import base64
import json
import logging
import os
from pathlib import Path
from collections import deque  # 这个是时间窗口的包
from minio import Minio
from minio.deleteobjects import DeleteObject
from config.llm_config import lm_config
from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
import re
from typing import List, Tuple, Dict, Deque
import time
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """
    name = "node_md_img"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行......")

        # 1，参数处理
        md_content,md_path_obj,images_dir = self._step_1_get_content(state)
        print(f"md_content:{md_content},md_path_obj:{md_path_obj},images_dir:{images_dir}")

        # 2，图片扫描
        target_images = self._step_2_scan_images(md_content,images_dir)
        print(f"target_images:{target_images}")

        # 3,视觉模型摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem,target_images)

        # 4,上传minio，替换md                              # 文档名称        图片的信息       摘要       原始文档
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem,target_images,summaries,md_content)

        # 5，保存备份
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"],new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state
    # 步骤一
    def _step_1_get_content(self, state):
        # 1，校验参数
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name="md_path",expected=str)
        md_path_obj = Path(md_path)

        if not md_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在,{md_path}")

        md_content = state["md_content"]

        # 测试用代码
        if not md_content:
            with open(md_path,"r",encoding="utf-8") as md_file:
                md_content = md_file.read()

        # 2，图片路径 注意这里加一个parent，因为上一个节点返回的路径是md文档的路径，加一个parent就是返回他的上一级目录
        images_dir = md_path_obj.parent / "images"
        return md_content,md_path_obj,images_dir
    # 步骤二
    def _step_2_scan_images(self, md_content,images_dir):

        # 1，返回结果
        target_images = []

        # 2，遍历images_dir，取出图片
                    # {images_dir}:所有图片所在的路径
        for image_file in os.listdir(images_dir):  # os.listdir:方法的作用是取出目标文件夹里面所有的文件和文件夹名称放到list中
            # splitext的作用是把文件的扩展名和文件路径切开
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:  # 如果文件的扩展名不在image_extensions中 则：
                self.logger.warning(f"图片格式不支持，跳过{image_file}")
                continue

            img_path = images_dir / image_file  # 图片路径
            context = self._find_image_in_md(md_content,image_file) # 找到图片的上下文 image_file:图片名

            # 过滤MD未引用的图片
            if not context:
                self.logger.warning(f"图片未在MD中引用，跳过处理{image_file}")
                continue

                                        # 下方元组内包含图片的名称，图片所在路径，上下文
            target_images.append((image_file,img_path,context))  # List([str,str,tuple(pro,post)])

        return target_images

    # 步骤二方法一
    def _find_image_in_md(self,md_content:str,image_file:str,context_len: int =100)->Tuple[str,str]:
        """
        找到图片截取上下文
        :param md_content:
        :param image_file:
        :param context_len:
        content_len：默认值100
        :return:
        """
        # 1、定义正则表达式
        # ![描述](images/文件名.扩展名)
        # r"字符串": 不要将其中的特殊符号进行转义
        # re.escape 转义图片文件名中的特殊字符，避免正则语法错误
        # .* 贪婪匹配  .*? 非贪婪匹配
        partern = re.compile(r"!\[.*?\]\((.*?)" + re.escape(image_file) + r".*?\)")
        match = partern.search(md_content)
        if not match:
            return None

        start , end = match.span()  # 取出正则匹配到的起始位置和结束位置

        pre_text = md_content[max(0,start - context_len):start]  # 文件上文
        post_text = md_content[end:min(len(md_content),end + context_len)] # 文件下文
        return pre_text,post_text

    # 步骤三 summaries:多份摘要
    def _step_3_generate_summaries(self,doc_stem: str,target_images: List[Tuple[str,str,Tuple[str,str]]]) -> Dict[str,str]:

        summaries = {}
        # 时间窗口限流
        request_deque = deque()  #：deque：是一个双端队列，专门用来在头尾添加或者删除元素

        for img_file,image_path,context in target_images:
            self._apply_api_rate_limit(request_deque,max_requests=10)
            # 调用视觉模型
            summaries[img_file] = self._summarize_image(image_path,root_folder=doc_stem,image_content=context) # 图片摘要
        return summaries

    # 步骤三的方法一
    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流

        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待: {sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数: {len(request_times)}")

    # 步骤三的方法二                    root_folder:是文件夹的意思，代表根目录
    def _summarize_image(self,image_path:str,root_folder:str,image_content:Tuple[str,str]) -> str:

        # 0,图片采用的是base64编码
        with open(image_path,"rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')


        # 1，vlm模型工具
        # ai = ChatOpenAI(model=lm_config.vl_model,
        #            temperature=lm_config.temperature,
        #            base_url=lm_config.base_url,
        #            api_key=lm_config.api_key,
        #            model_kwargs={"response_format":{"type":"json_object"}})
        vl_ai = get_llm_client(lm_config.vl_model)

        # 2，调用模型
        message = [
            {
                "role":"user",
                "content":[
    {
        "type": "text",    # 这里的root_folder文件名
        "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。尽量不要超过10个字"""
    },
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        }
    }
]
            }
        ]
        response = vl_ai.invoke(message)

        # 3，处理返回的摘要信息
        return response.content.strip().replace("\n","")

    def _step_4_upload_and_replace(self,doc_stem:str,target_images:List[Tuple[str,str,Tuple[str,str]]]
                                   ,summaries:Dict[str,str],md_content:str):
        # 0,minio的客户端
        minio_client = get_minio_client()
        minio_img_dir = minio_config.img_dir
        # {minio_img_dir}:桶名  {doc_stem}:原始的pdf文件的名称不带后缀的 见：test4
        upload_dir = f"{minio_img_dir}/{doc_stem}"
        print(f"将图片存入{upload_dir}目录下")

        # 1，清理minio目录
        self.clean_minio_directory(minio_client,upload_dir)

        # 2，批量上传图片，获得minio的urls
        urls = self.upload_images_batch(minio_client,upload_dir,target_images)

        # 3，将摘要和url路径合并
        image_info = self.merge_summary_and_url(summaries,urls)

        # 4，替换md文件的摘要和路径
        md_content = self.process_md_file(md_content,image_info)

        return md_content

    # 步骤四方法一：清理目录
    def clean_minio_directory(self, minio_client:Minio, upload_dir):
        objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=upload_dir, recursive=True)
        # 构造删除列表
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
        if delete_list:
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for error in errors:
                self.logger.error(f"删除失败: {error}")

    # 步骤四的方法二 批量上传文件
    def upload_images_batch(self,minio_client:Minio,upload_dir:str,target_images:List[Tuple[str,str,Tuple[str,str]]]):
        urls = {}
        for img_file,img_path, _ in target_images:
            # 上传
            object_name = f"{upload_dir}/{img_file}"  # minio文件的对象名（路径：带后缀）
            print(f"上传文件{object_name}")
            urls[img_file] = self.upload_to_minio(minio_client,img_path,object_name)  # 返回minio的url

        return urls

    # 步骤四的方法三 合并参数
    def merge_summary_and_url(self,summaries:Dict[str,str],urls) -> Dict[str,Tuple[str,str]]:
        image_info = {}
        for image_file , summary in summaries.items():
            # image_info[image_file] = (summary,urls[image_file])  # 方法一
            if url := urls.get(image_file):   #方法二
                # 在image_file中找url找到就赋值
                image_info[image_file] = (summary,url) # 存储摘要和访问链接
        return image_info

    # 步骤四的方法四 替换md文档中的urls和摘要
    def process_md_file(self,md_content:str,image_info:Dict[str,Tuple[str,str]]):
        for image_file,(summary,url) in image_info.items():
            # ?号的正则中的非贪婪匹配，re.escape:是去掉特殊字符的字符串
            pattern = re.compile(r"!\[.*?\]\(.*?"+re.escape(image_file)+r".*?\)")
            md_content = pattern.sub(lambda m: f"![{summary}]({url})",md_content)
        return md_content


    # 步骤四的方法五 上传minio
    def upload_to_minio(self,minio_client:Minio,img_path:str,object_name:str) -> str:

        # 上传minio 调用fput_object:把本地磁盘上的一个文件上传到minio对象存储
        ifSuccess = minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=img_path,
            content_type=f"image/{os.path.splitext(img_path)[1][1:0]}")
        print(f"上传结果{ifSuccess}")
                                                                # object_name：项目名（kb01）/文件名（带后缀）
        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"                               
        print(f"图片url：{url}")
                                                                       
        return url
    # 步骤5保存和备份文件
    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """
        # 构造新文件路径：替换原后缀为 _new.md os.path.solitext:将文件拆成不带后缀的路径名和文件后缀，这里取得第0个元素就是不带后缀的路径名称
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        # 这里的new_md_file_name就是替换过路径的新的文件

        # 写入新MD内容（覆盖写入，若文件已存在则更新）
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.logger.info(f"处理后MD文件已保存，新文件路径: {new_md_file_name}")

        return new_md_file_name

import json
from typing import List, Dict
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from processor.import_processor import get_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """
    name = "node_document_split"

    def process(self, state: ImportGraphState):

        # 1，参数处理  file_title:在节点a处有取出
        content,file_title = self._step_1_get_input(state)

        # 2，标题切（初切）  sections[切出来的文件块]   title_count(标题数等于0的的话触发无标题兜底)  lines_count(文档中的行)
        sections , title_count , lines_count= self._step_2_split_by_title(content,file_title)

        print("-----------------------------------------------------------------------------------")

        # 3，无标题兜底（默认标题）
        sections = self._step_3_handle_no_title(content,sections,title_count, file_title)

        # 4，块的精细化处理
        sections = self._step_4_refine_chunks(sections)
        for section in sections:
            print(f"{section['title']}")
            print(f"{section['content']}")
            print("=========================================================================")
        print(f"标题数量:{title_count}")
        print(f"行数：{lines_count}")

        # 5，打印日志
        self._step_5_print_stats(lines_count,sections)

        # 6，备份
        self._step_6_backup(state,sections)

        state["chunks"] = sections
        return state

    # 步骤一:参数处理
    def _step_1_get_input(self,state):
        print("node_document_split:步骤1：参数处理")

        # 非空校验
        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name = "md_content",expected_type=str)

        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name = "file_title",expected_type=str)

        # 2、基础标准化：统一换行符
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        return content,file_title

    # 步骤二：标题切
    def _step_2_split_by_title(self,content,file_title):
        print("node_document_split:步骤2：标题切(初切)")

        # 参数声明   sections：拆分出来的段落
        sections: List[Dict[str,str]] = []
        title_count = 0
        # 初切的时候按转换符切
        lines = content.split("\n")
        # 存放代码快的地方
        current_lines = []
        # 声明一个变量，判断取出来的到底是标题还是别的#
        in_code_block = False
        current_title = ""  # current:表示当前的意思

        # 切块逻辑（标题切）
        title_pattern = r'\s*#{1,6}\s+.+'  # 标题的正则

        # 独立封装刷新块的逻辑函数
        def _flush_sections():
            # current_lines:为空就是两个标题挨着了
            if not current_lines:
                return
            # 封装sections快的案例代码
            sections.append(
                {
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "parent": "",
                    "file_title": file_title,
                }
            )

        # 遍历文档判断是否是代码块还是标题，或者其他的#号
        # 也就是循环文档中用\n拆分的
        for line in lines:
            striped_line = line.strip() # strip:删除这一行开头和结尾的空格，换行，Tab

            # 判断是否在代码块中
            if striped_line.startswith("```") or striped_line.startswith("~~~"):
                # 第一次碰见```的时候：in_code_block = True,第二次碰见代表走出代码块，变回false
                in_code_block = not in_code_block
                current_lines.append(line)   # 存放到：current_lines中
                continue
            # 判断它是个标题且不在代码块中
            if (not in_code_block) and (re.match(title_pattern,line)):
                # 标题处理，将之前存好的块的内容封装成一个sections块
                _flush_sections()
                current_title = striped_line  # 换标题
                # current_lines = []
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(line)  # 普通行或者代码块

        _flush_sections()
        print(f"标题初切：sections:{sections}")
        return sections,title_count ,len(lines)

    # 步骤三：无标题兜底(默认标题)
    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        print("node_document_split:步骤3：无标题兜底(默认标题)")
        if title_count == 0:
            return [{"title":"无标题","content":content,"file_title":file_title}]
        return sections

    # 步骤四：精细化拆分(长切短和)
    def _step_4_refine_chunks(self, sections):
        print("node_document_split:步骤四：精细化拆分(长切短和)")

        # 长切列表
        refined_split = []
        for sec in sections:
            refined_split.extend(self.split_long_section(sec))  # 长切操作
            # extend:把传入列表里面的每一项，挨个拿出来，追加到当前列表尾部

        # 短合列表
        final_sections = self.merge_short_sections(refined_split) # 短合操作

        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections

    # 步骤五，打印日志
    def _step_5_print_stats(self, lines_count, sections):
        print("node_document_split:步骤5，打印日志")
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数: {lines_count}")
        self.logger.info(f"最终生成Chunk数量: {chunk_num}")

    # 6，备份
    def _step_6_backup(self, state, sections):
        print("node_document_split:步骤6，备份")

        # 将sections切分结果输出的文档路径 下方.parent的意思就是父级，父母的意思
        path = Path(state.get("md_path")).parent / f'{Path(state.get("file_title")).stem}_new_chunks.json'
        with open(path,"w",encoding="utf-8") as f:
            json.dump(
                sections,
                f,
                ensure_ascii=False,   # 这个参数得意思是，如果有中文得话显示出来，不要转为别的编码格式
                indent=2,             # 这个参数是设置的缩进，首行缩进
            )

    # 步骤四的方法一长切
    def split_long_section(self, section: Dict[str,str]):
        print("node_document+split:步骤4的方法1长切")
        content = section.get("content","")
        content_len = len(content)
        # 长度合标，直接返回
        if content_len <= get_config().max_content_length:
            return [section]

        title = section.get("title")  # 没有换行符的title
        prefix = f"{title}\n\n" if title else ""
        available_len = get_config().max_content_length - len(prefix)  # 切分标准

        # 极端情况：标题长度超过阈值，无法切分，返回原章节
        if available_len <= 0:
            self.logger.warning(f"章节标题过长，无法切分: {title[:20]}...")
            return [section]

        # 去重标题
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()  # lstrip: 去掉左侧的空行

        # 切分 -> langchain的递归文档切分器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = available_len,   # 这个参数是设置标准切分大小
            chunk_overlap = 0,   # 这个是参数是向量化文本的时候覆盖的设置（作用是防止丢失），切分的时候先不设置
            separators = ["\n\n", "\n", "。", "，", "；", "！", "？", " "]

        )

        # 切分结果
        sub_sections = []

        for index , chunk in enumerate(splitter.split_text(body),start=1):
            text = chunk.strip()
            if not text:
                continue
            full_text = (prefix + text).strip()

            sub_sections.append(
                {
                    "title": f"{title} - {index}" if title else f"chunk - {index}",
                    "content":full_text,
                    "parent_title":title,
                    "part": index,  # 长切时产生的，表示是该章节第几个子块
                    "file_title": section.get("file_title")
                }
            )

        return sub_sections

    # 步骤四的方法二短合
    def merge_short_sections(self, sections):
        print("node_document+split:步骤4的方法2短合")

        # 小于阈值 + 合前一个章节同父标题才可以合
        if not sections:
            self.logger.debug("待合并chunk列表为空，直接返回")
            return []

        merged_sections = []  # 最终合并结果
        current_chunk = None  # 迭代累加器：保存当前待合并的Chunk
        # sections：代表的是所有的块 sec：是下一个章节的内容 content_chunk：是当前章节的内容
        for sec in sections:
            # 初始化：第一个Chunk直接作为当前待合并块
            if current_chunk is None:
                current_chunk = sec
                continue

            # 合并条件：1.当前块长度不足阈值 2.与下一块同父标题（同属一个原章节）
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                # 合并前清理：去掉下一块开头重复的父标题，避免内容冗余
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):].lstrip()
                # 合并内容：空行分隔，保证格式整洁
                current_chunk["content"] += "\n\n" + next_content
                # 更新子chunk序号：保留最新序号，便于溯源
                if "part" in sec:
                    current_chunk["part"] = sec["part"]
                self.logger.debug(
                    f"合并短Chunk: {current_chunk.get('parent_title')} - 累计长度{len(current_chunk['content'])}")
            else:
                # 不满足合并条件：将当前块加入结果，切换为新的待合并块
                merged_sections.append(current_chunk)
                current_chunk = sec

                # 循环结束后，将最后一个待合并块加入结果
        if current_chunk is not None:
            merged_sections.append(current_chunk)

            self.logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
        return merged_sections

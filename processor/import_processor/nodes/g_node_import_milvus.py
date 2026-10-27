import logging
from typing import Dict, Any
from pymilvus import DataType
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
import json

from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化（建立表，幂等性）
    """
    name = "node_import_milvus"

    def process(self, state: ImportGraphState):

        # 1，数据校验 vector_dimension：数据维度
        chunks_json_data,vector_dimension = self._step_1_check_inputs(state)

        # 2，客户端和集合名准备
        milvus_client = self._step_2_prepare_collection(vector_dimension)

        # 3，清理可能的冗余数据（幂等性）
        self._step_3_clean_old_data(milvus_client,chunks_json_data)

        # 4，数据入库（字段和索引）,返回数据库主键
        update_chunks = self._step_4_insert_data(milvus_client,chunks_json_data)
        # 下面是打印的向量主键id的代码
        # for chunk in update_chunks:
        #     print(f"chunks_id:{chunk['chunks_id']}")

        # 这里也可以参照前一个文档生成新的文档

        # 5，更行状态
        state["chunks"] = update_chunks

        return state

    # 步骤一：参数校验
    def _step_1_check_inputs(self, state:Dict[str,Any]):
        print(f"node_import_milvus:步骤1，数据校验")
        chunks_json_data = state.get("chunks")
        if not chunks_json_data:
            raise StateFieldError(field_name="chunks",message="chunks不能为空",expected_type=list)

        # 向量维度
        dimenstion = len(chunks_json_data[0]["dense_vector"])
        print(f"向量维度:{dimenstion}")
        # 这里不能写死，embedding模型不同生成的向量维度不同，所以不能写死，换 embedding 模型时，kb_item_names 会直接建表失败或维度不匹配
        return chunks_json_data,dimenstion

    # 步骤二：准备客户端和集合（字段和索引）
    def _step_2_prepare_collection(self, vector_dimension:int):
        print(f"node_import_milvus:步骤2，结构准备")
        # 1，客服端
        milvus_client = get_milvus_client()

        # 2，集合
        collection_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collection_name):
            self.create_chunks_collection(milvus_client,collection_name,vector_dimension)

        return milvus_client

    # 步骤三：清理冗余数据（幂等性）
    def _step_3_clean_old_data(self, client, chunks_json_data):
        print(f"node_import_milvus:步骤3，清理旧数据")
        # 1，获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2,执行幂函数清理
        self.clear_chunks_by_file_title(client ,file_title)

    # 步骤四：批量入库返回updata_chunks主键id
    def _step_4_insert_data(self,milvus_client, chunks_json_data:list[Dict[str,Any]]):
        print(f"node_import_milvus:步骤4，插入数据")

        # 1，数据处理
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()  # 深拷贝源数据，判断part字段是否为空，为空赋值为0
            if "part" not in item_copy:
                item_copy["part"] = 0
            data_to_insert.append(item_copy)
            # 将插入的循环出来的结果放到新的容器中

        # 2，批量插入
        insert_result = milvus_client.insert(collection_name=milvus_config.chunks_collection,data=data_to_insert)
        insert_conut = insert_result.get("insert_conut")  # 插入条数
        print(f"插入数据条数:{insert_conut}")

        # 3，回写主键id
        insert_ids = insert_result.get("ids")
        for index,item in enumerate(chunks_json_data):
            item["chunks_id"] = str(insert_ids[index])

        # 返回带有主键的chunks
        return chunks_json_data

    # 步骤二方法一
    def create_chunks_collection(self, milvus_client, collection_name, vector_dimension):
        print(f"node_import_milvus,步骤2：创建集合{collection_name}")
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 字段声明
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

        # 索引声明
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
            )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="decse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # 倒排索引
            metric_type="IP",
            params={"inverted_index_algo":"DAAT_MAXSCORE","normalize":True,"quantization":"none"}
        )

        # 创建集合
        milvus_client.create_collection(collection_name=collection_name, schema=schema,index_params=index_params)

    # 步骤3方法1
    def clear_chunks_by_file_title(self, client, file_title):
        print("node_import_milvus,步骤3方法1:清理旧数据")
        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name = milvus_config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败:{str(e)}")
            raise MemoryError(f"Milvus 数据删除失败:{str(e)}")

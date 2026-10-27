import json

from config.milvus_config import milvus_config
from processor.query_porcessor.base import NodeBase
from processor.query_porcessor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑")
        # 1，参数处理
        item_names = state["item_names"]  # 原条件过滤
        query = state["rewritten_query"]  # 语义搜索条件

        query_embeddings = generate_embeddings([query]) # 参数向量化
        dense_vector = query_embeddings["dense"][0]
        sparse_vector = query_embeddings["sparse"][0]

        # 2，milvus客户端
        milvus_client = get_milvus_client()  # 客户端
        chunks_collection = milvus_config.chunks_collection  # 表（集合）

        # 3，请求对象
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=f'item_name in ({item_names})'  # 这里要用单引号 milvus对单双引号非常敏感
        )
        # 4,发送请求
        response = hybrid_search(
            client=milvus_client,
            collection_name=chunks_collection,
            reqs=reqs,
            ranker_weights=(0.8,0.2),  # 稠密向量和稀疏向量得配比
            output_fields=["chunks_id","content","item_name"]  # 要查询输出得字段
        )

        print(response[0])
        # return state  这个三元表达式得意思是如果response不为空则拿response[0] 如果为空返回[]
        return {"embedding_chunks": response[0] if response else []}  # [[]] 结果解析

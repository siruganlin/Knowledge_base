import json
from config.milvus_config import milvus_config
from processor.query_porcessor.base import NodeBase
from processor.query_porcessor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_porcessor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能: HyDE（Hypothetical Document Embedding）
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    防止源数据过滤的时候把真是需要的数据过滤掉了
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1，参数处理
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")

        # 2，大模型生成假设性文档
        hyde_doc = self._step_1_create_embedding_hyde_doc(rewritten_query)
        print(f"假设性回答：{hyde_doc}")

        # 3，在假设性文档得基础之上，用“重写问题”和“假设性文档”搜索文本块
        response = self._step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names
        )

        # 4，结果解析
        # json_response = json.dumps(response, ensure_ascii=False, indent=4)
        print(f"response[0]:{response[0]}")
        # return state
        return {"hyde_embedding_chunks": response[0] if response else [],"hyde_doc": hyde_doc}

    # 步骤1
    def _step_1_create_embedding_hyde_doc(self, rewritten_query: str) -> str:
        print("步骤1：基于大模型生成假设性文档")

        # 1,客户端
        ai_client = get_llm_client()

        # 2，提示词
        hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)

        # 3,假设性搜索
        hyde_doc = ai_client.invoke(hyde_prompt).content

        return hyde_doc

    # 步骤2
    def _step_2_search_embedding_hyde(self, rewritten_query: str, hyde_doc, item_names: list):
        print("步骤2：根据假设性文档进行向量搜索")

        # 1,拼接上下文
        embedding_context = rewritten_query + "" + hyde_doc

        # 2,将上下文向量化
        embeddings = generate_embeddings([embedding_context])
        dense_vector = embeddings.get("dense")[0]
        sparse_vector = embeddings.get("sparse")[0]

        # 3,向量搜索
        # 客户端
        milvus_client = get_milvus_client()
        collection_name = milvus_config.chunks_collection

        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=f'item_name in ({item_names})'
        )

        response = hybrid_search(
            client=milvus_client,
            collection_name=collection_name,
            reqs=reqs,
            limit=5,
            ranker_weights=(0.8, 0.2),
            output_fields=["chunks_id","content","item_name"]
        )

        return response

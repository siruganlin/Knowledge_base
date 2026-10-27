import json
from typing import List, Tuple

from processor.query_porcessor.base import NodeBase
from processor.query_porcessor.state import QueryGraphState
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能: Reciprocal Rank Fusion
    将多路径召回的结果（向量、HyDE、Web）进行加权融合排序。
    rrf算法：倒数排名融合
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1，参数校验
        embedding_chunks = state.get("embedding_chunks")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")
        embedding_chunks_list = [doc.get("entity") for doc in embedding_chunks ]
        hyde_embedding_chunks_list = [doc.get("entity") for doc in hyde_embedding_chunks]

        # 2，封装要融合的数据
        rrf_inputs = [
            (embedding_chunks_list,1.0),
            (hyde_embedding_chunks_list,1.0)
        ]

        # 3，使用rrf算法公式对融合对象进行融合
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        # 4，放回结果处理
        rrf_chunks = [doc for doc, _ in rrf_merge_results]  # 只要文档不要分
        state["rrf_chunks"] = rrf_chunks
        print(f"rrf_chunks: {rrf_chunks}")
        return state

    def _rrf_merge(self, rrf_inputs:List[Tuple],k:int=60,max_results:int = 5) -> List[Tuple]:
        print("向量搜索和假设性搜索的融合函数")
        chunk_scores = {}
        chunk_data = {}

        # 循环处理融合数据
        for rrf_input , weight in rrf_inputs: # 取出的是每一组的检索结果和权重
            for rank , doc in enumerate(rrf_input): # 读取这组里每一条文档，rank是它在本组里的排名
                chunk_id = doc.get("chunks_id")
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank) # RRF 算分。排名越靠前，加分越多；同一个文档多次出现，分数叠加
                chunk_data.setdefault(chunk_id,doc) # 保存文档内容，同一个文档只存一次，实现去重

        # 未排序的结果 score:是分数 cid：是id
        unsorted_results = [(chunk_data[cid],score) for cid , score in chunk_scores.items()]  # List[Tuple(doc,score)]

        # 排序结果
        sorted_results = sorted(
            unsorted_results,
            key=lambda x: x[1],
            reverse=True
        )

        # 返回rrm排序结果
        return sorted_results[:max_results] if max_results else sorted_results

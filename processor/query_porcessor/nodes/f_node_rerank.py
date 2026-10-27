import json
from typing import Dict, List, Any

from processor.query_porcessor.base import NodeBase
from processor.query_porcessor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能: 使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    把 query 和每段正文一起喂给模型精确打分，再按分排序
    和上一个节点最大的区别就是：
    e：只按分数重排序，合并
    f：调用Cross-Encoder 模型来对每一段正文进行精细化的相关性评分
    两者结合才能达到低成本 + 高精度的检索排序
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 1，将web_doc和rrf_chunks节点的数据合并
        merged_multi_cocs:List[Dict[str,Any]] = self._step_1_multi_source_docs(state)
        print("merged_multi_cocs:", merged_multi_cocs)

        # 2，调用排序模型，对合并的结果进行排序
        reranked_docs:List[Dict[str,Any]] = self._step_2_rerank_docs(state,merged_multi_cocs)
        """
        排序和前面rrf和区别：rrf只看排名位置，重排序模型真的把query和每一段正文一起读进模型打分，
        所以精度高但是慢，这就是为什么要rrf粗步筛（5条）再重排精算
        """
        # 3，断崖测试（分数擦差距，分数比率）去掉过低的匹配结果
        cutoff_docs:List[Dict[str,Any]] = self._step_3_cliff_cutoff(reranked_docs)

        # 4，放回结果封装
        print("步骤4：结果封装")
        state["reranked_docs"] = cutoff_docs
        print(cutoff_docs)
        return state

    # 步骤1：将web_doc和rrf_chunks节点的数据合并
    def _step_1_multi_source_docs(self, state):
        print("步骤1：将web_doc和rrf_chunks节点的数据合并")
        final_docs = []
        rrf_chunks = state.get("rrf_chunks")
        web_docs = state.get("web_search_docs")
        for rrf_doc in rrf_chunks:
            format_rrf_doc = {
                "content":rrf_doc.get("content"),
                "title":rrf_doc.get("title"),
                "chunk_id":rrf_doc.get("chunks_id"),
                "url":None,
                "source":"local"
            }
            final_docs.append(format_rrf_doc)
        for web_doc in web_docs:
            format_web_doc = {
                "content": web_doc.get("snippet"),
                "title": web_doc.get("title"),
                "chunk_id": None,
                "url": web_doc.get("url"),
                "source": "web"
            }
            final_docs.append(format_web_doc)

        return final_docs

    # 步骤2：调用排序模型对合并后的结果进行排序
    def _step_2_rerank_docs(self, state, merged_multi_cocs):
        print("步骤2：调用排序模型对合并后的结果进行排序")
        rewritten_query = state.get("rewritten_query")
        contents = [doc.get("content") for doc in merged_multi_cocs]
        # 调用排序模型根据rewritten_query对contents进行排序
        rerank_scores = rerank_documents(rewritten_query, contents)
        scored_docs = [{**doc, "score":score} for doc, score in zip(merged_multi_cocs, rerank_scores)]
        # zip:把两个list中位置相同的两个元素配对在一起 **doc：把doc所有的键值对展开，额外加上这个新的key（score），形成一个全新的字典
        # 两种方式
        # scored_docs = []
        # for doc, score in zip(merged_multi_cocs, rerank_scores):
        #     scored_docs.append({
        #         "contnet":doc.get("content"),
        #         "title":doc.get("title"),
        #         "chunk_id":doc.get("chunk_id"),
        #         "url":doc.get("url"),
        #         "source":doc.get("source"),
        #         "score":float(score)
        #
        #     })

        # 排序
        scored_socre_docs = sorted( # sorted:用来排序列表，生成新的列表
            scored_docs,  # 做排序的文档
            key=lambda x: x["score"], # 那文档中的每个score来进行排序
            reverse=True # 从大到小
        )
        # print("scored_docs_docs:", scored_socre_docs)
        return scored_socre_docs

    # 步骤3：断崖测试（分数差距，分数比率）去过低的匹配结果
    def _step_3_cliff_cutoff(self, reranked_docs):
        print("步骤3：断崖测试（分数差距，分数比率）去过低的匹配结果")
        # 最多要多少条结果
        upper_bound = min(10,len(reranked_docs))
        # 最少要多少结果
        lower_bound = min(3,len(reranked_docs))
        # 断崖位置
        cutoff_pos = upper_bound
        for index in range(lower_bound - 1, upper_bound - 1):
            current_score = reranked_docs[index].get("score")
            next_score = reranked_docs[index + 1].get("score")
            # 分差
            abs_gap = current_score - next_score
            # 比率 下面的处理是为了避免分母为零1e-6为10的负6次方
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= 0.5 or rel_gap >=0.25:
                cutoff_pos = index + 1  # 找到断崖位置
                break
        return reranked_docs[:cutoff_pos]
"""
断崖的两个指标（分差和比率）任意一个达标（abs_gap ≥ 0.5 或 rel_gap ≥ 0.25）就判定为断崖、在那里收口
断崖的含义：保留断崖的那一条，丢掉它后面的全部
遍历是从 lower-1（第 3 条）开始的，所以前 3 条永远保留（至少 3 条、至多 10 条）
"""

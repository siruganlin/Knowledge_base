from typing import List

from dotenv import load_dotenv

from config.rerankere_config import reranker_config

load_dotenv()
import dashscope
def rerank_documents(query:str,documents: List[str]) -> List[float]:
    text_rerank_api_key = reranker_config.text_rerank_api_key # 百炼通用ker
    text_rerank_model = reranker_config.text_rerank_model

    # 调用排序模型对文档相关度打分
    response = dashscope.TextReRank.call(
        model=text_rerank_model,
        api_key=text_rerank_api_key,
        query=query,
        documents=documents,
        instruct=reranker_config.text_rerank_instruct,
        top_n=len(documents)
    )
    # print(response)

    status_code = response.get("status_code")
    if status_code != 200:
        message = response.get("message")
        raise RuntimeError(f"Dashscope rerank 调用失败: {message}")
    # 查看数据结构
    reranke_scores = [0.0] * len(documents)
    results = response.get("output").get("results")
    for index,result in enumerate(results):
        reranke_scores[index] = result.get("relevance_score")



    print(f"reranke_scores,{reranke_scores}")
    return reranke_scores



    # return scores

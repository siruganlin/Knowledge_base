import asyncio
import json
from agents.mcp import MCPServerStreamableHttp
from config.bailian_mcp_config import mcp_config
from processor.query_porcessor.base import NodeBase
from processor.query_porcessor.state import QueryGraphState
from tool.logger import logger


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # TODO
        logger.info(f"【{self.name}】节点逻辑")
        # 1，参数
        query = state.get("rewritten_query")  # 问题
        docs = [] # 结果

        # 2，调用外部搜索引擎
        result = asyncio.run(self._mcp_call(query))

        # 3,解析放回结果
        json_text = result.content[0].text
        result_json_obj = json.loads(json_text)
        pages = result_json_obj.get("pages")
        for page in pages:
            snippet = (page.get("snippet") or "").strip()
            url = (page.get("url") or "").strip()
            title = (page.get("title") or "").strip()
            docs.append({"snippet":snippet, "title":title, "url":url})

        # 4，return state
        print(docs)
        # 这个return一个字典式的数据，会自动添加到state全局状态中这是langgraph中的一个特性
        return {"web_search_docs": docs}

    async def _mcp_call(self, query):
        search_mcp_tool = MCPServerStreamableHttp(
            name = "search_mcp_tool",
            params = {
                "url":mcp_config.mcp_base_url,
                "headers":{"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout":10
            },
            cache_tools_list=True, # 开启缓存
            max_retry_attempts=3  # 最大尝试次数
        )

        # 连接服务器
        await search_mcp_tool.connect()
        result = await search_mcp_tool.call_tool(
            tool_name="bailian_web_search",
            arguments={"query":query,"count":3},
        )

        await search_mcp_tool.cleanup() # 刷新清理缓存

        return result

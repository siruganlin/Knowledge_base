# 安排每个节点
import logging
from langgraph.graph import StateGraph, START, END
from processor.import_processor import ImportGraphState
from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.a_node_entry import NodeEntry
from processor.import_processor.nodes.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.nodes.c_node_md_img import NodeMDImg
from processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.e_node_item_name_recongnition import NodeItemNameRecognition
from processor.import_processor.nodes.f_node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus



class KBImportWorkflow():

    def __init__(self,config=None):
        self._compiled_graph = None


    @property
    def graph(self):
        """
            返回图实例:return:
        """
        logging.info("获取图实例")
        if self._compiled_graph is None:
            self._compiled_graph = self.builder_graph()  #创建图

        return self._compiled_graph

    """
    工具方法（静态方法，写这个就可以不用写self了）
    这里下方到antry就不会执行了，因为这个路由方法要等着，
    入口节点执行完之后返回一个state值才能if判断，如果入口节点什么都还没写的话，
    就会走END那条路线直接退出了
    """
    @staticmethod
    def route_after_entry(state:ImportGraphState):
        if state.get("is_pdf_read_enabled"):
            return "b_node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "c_node_md_img"
        else:
            logging.info(f"route_after_entry路由器：未指定导入文件类型")
            return END

    def builder_graph(self):
        """
        创建主图
        :return:
        """
        graph = StateGraph(ImportGraphState)

        # 1，注册节点（传实例，LangGraph 才会调用 __call__ → 打印日志 → 执行 process）
        graph.add_node("a_node_entry", NodeEntry())
        graph.add_node("b_node_pdf_to_md", NodePDFToMD())
        graph.add_node("c_node_md_img", NodeMDImg())
        graph.add_node("d_node_document_split", NodeDocumentSplit())
        graph.add_node("e_node_item_name_recongnition", NodeItemNameRecognition())
        graph.add_node("f_node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus", NodeImportMilvus())

        graph.set_entry_point("a_node_entry")
        #2，定义节点边
        # 这里定义一条条件边
        graph.add_conditional_edges(
            "a_node_entry",  # 在这个节点跑完之后出现分支
            self.route_after_entry,
            {
                "c_node_md_img":"c_node_md_img",  # 这一段的意思是看定义的函数放回的值从而，判断往哪个节点走
                "b_node_pdf_to_md":"b_node_pdf_to_md",
                END:END
            }
        )

        graph.add_edge("b_node_pdf_to_md","c_node_md_img")
        graph.add_edge("c_node_md_img","d_node_document_split")
        graph.add_edge("d_node_document_split","e_node_item_name_recongnition")
        graph.add_edge("e_node_item_name_recongnition","f_node_bge_embedding")
        graph.add_edge("f_node_bge_embedding","g_node_import_milvus")


        #3， 编译图
        graph_compile = graph.compile()
        return graph.compile()

    def run(self,state:ImportGraphState,stream:bool=False):

        if stream:
            return self.graph.stream(state,output_mode="values")
        else:
            return self.graph.invoke(state)

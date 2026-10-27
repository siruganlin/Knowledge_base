import logging
from pathlib import Path
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from processor.import_processor.base import BaseNode

class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """
    name = "node_entry"

    def process(self, state: ImportGraphState):

        logging.info(f"{self.name}节点开始执行......")
        # 1，获得输入路径
        import_file_path = state.get("import_file_path")

        # 2，校验路径存在
        if not import_file_path:
            raise StateFieldError(field_name="import_file_path",expected_type=str)

        # 3，4 校验文件
        # Path是python的一个对象，可以校验文件路径
        import_file_path_obj = Path(import_file_path)
        if  not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件不存在：{import_file_path_obj}")

        # 5，判断是md还是pdf
        # Path有一个方法，suffix：可以判断后缀名
        if import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        elif import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        else:
            raise FileProcessingError(message=f"不支持的文件格式:{import_file_path_obj}"
                                      )
        state["file_title"] = import_file_path_obj.stem # .stem的作用是拿文件名，去掉后缀
        state["file_dir"] = r"D:\output"  # file_dir：输出保存目录
        return state


from typing import TypedDict, List, Optional
import copy


class ImportGraphState(TypedDict, total=False):
    """
    导入流程状态：
    记录从文件接入、PDF解析、Markdown处理、文档切分、向量化到向量库导入的全过程数据。
    """

    # ====================== 任务标识与控制开关 ======================
    task_id: str  # 任务 ID，用于任务追踪与日志关联
    is_md_read_enabled: bool  # 是否启用 MD 读取
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    # ====================== 文件路径信息 ======================
    import_file_path: str  # 导入文件路径
    file_dir: str  # 导入/输出文件目录
    pdf_path: str  # PDF 文件路径
    md_path: str  # 转换后 Markdown 文件路径

    # ====================== 文件元信息 ======================
    file_title: str  # 文件标题（不含扩展名）
    item_name: str  # 识别出的商品/产品名称（如有）

    # ====================== 文档内容 ======================
    md_content: str  # Markdown 文档内容
    chunks: List[str]  # 文档切分后的文本片段列表


# ====================== 默认状态 ======================
GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_ad_read_enabled": False,
    "is_pdf_read_enabled": False,
    "import_file_path": "",
    "file_dir": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "item_name": "",
    "md_content": "",
    "chunks": [],
}


def create_default_state(overrides: Optional[ImportGraphState] = None) -> ImportGraphState:
    """
    创建默认状态，并允许覆盖部分字段。

    Args:
        overrides: 需要覆盖的字段。

    Returns:
        初始化后的 ImportGraphState。

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="xxx")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)

    if overrides:
        state.update(overrides)

    return state


def get_default_state() -> ImportGraphState:
    """获取一份默认状态副本。"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)

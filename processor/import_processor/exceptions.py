"""
base_node.py
LangGraph导入流程基础抽象节点、日志工具、异常定义
"""
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

T = TypeVar("T")


class ImportProcessError(Exception):
    """导入处理流程的基础异常"""
    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)


class StateFieldError(ImportProcessError):
    """状态字段错误。
    从 state 中获取必需字段缺失、为空或类型不符时抛出。

    Attributes:
        field_name: 缺失或无效的字段名称。
        expected_type: 期望的字段类型（可选）。
    """
    def __init__(
        self,
        node_name: str = "",
        field_name: str = "",
        expected_type: type = None,
        message: str = "",
        cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f", 期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class TypeMismatchError(ImportProcessError):
    """类型不匹配，节点输入输出类型不符合预期"""
    pass


class ConfigurationError(ImportProcessError):
    """配置错误，环境或参数配置无效"""
    pass


class FileProcessingError(ImportProcessError):
    """文件处理错误，文件不存在、格式错误、读写出错"""
    pass


class PdfConversionError(FileProcessingError):
    """PDF解析错误，MinerU调用失败"""
    pass


class ImageProcessingError(FileProcessingError):
    """图片处理错误"""
    pass


class DocumentSplitError(ImportProcessError):
    """文档切分错误：切分逻辑异常"""
    pass


class EmbeddingError(ImportProcessError):
    """向量化错误：模型调用失败、向量生成异常"""
    pass


class LLMError(ImportProcessError):
    """LLM 调用错误：API 调用失败、响应解析失败"""
    pass


# ===================== 基础抽象节点 =====================
class BaseNode(ABC, Generic[T]):
    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        # self.config = config or get_config()  # 你项目的全局配置，按需自行补全
        self.config = config
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口
        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        try:
            # 1. 开始准备执行节点
            self.logger.info(f"--- {self.name} 开始 ---")

            # 2. 执行节点
            result = self.process(state)

            # 3. 执行节点成功
            self.logger.info(f"--- {self.name} 完成 ---")
            return result

        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑
        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# ===================== 日志工具 =====================
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

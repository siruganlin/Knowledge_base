"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from .import_config import ImportConfig, get_config
from .exceptions import ImportProcessError
# from knowledge.utils.task_util import add_running_task, add_done_task  # 本地暂无 utils/task_util，先注释

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。
    """
    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import:{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Returns:
            更新后的状态实例

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        try:
            # 1，开始准备执行节点
            self.logger.info(f"--- {self.name} 开始 ---")

            # 2，执行节点
            result = self.process(state)

            # 3，执行节点成功
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
            message: 附加消息
        """
        log_msg = f"|{step_name}|"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

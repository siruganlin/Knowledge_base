"""
导入流程配置管理模块

集中管理导入流程所需的配置项，支持从 .env 文件加载环境变量。
工作流节点可通过 get_config() 获取全局单例，按需读取相应字段。

使用方式：
    from knowledge.processor.import_processor.config import get_config
    cfg = get_config()
    model = cfg.default_model
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Set
from dotenv import load_dotenv

# 模块加载时即读取项目根目录的 .env
load_dotenv()


# ====================== 环境变量读取辅助 ======================
def _env(key: str, default: str = "") -> str:
    """读取字符串环境变量"""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """读取整数环境变量，转换失败时回退默认值"""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    """读取布尔环境变量（支持 1/true/yes/on）"""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ImportConfig:
    """
    导入流程配置；所有字段均可由同名环境变量覆盖。
    BaseNode 在 __init__ 时会自动注入 get_config()。
    """

    # ===================== 文档处理配置 =====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200  # 图片上下文最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    overlap_sentences: int = 1  # 句子级切分时的重叠句数
    item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500  # 商品名识别时使用的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ====================== LLM（OpenAI 兼容接口） ======================
    openai_api_base: str = field(default_factory=lambda: _env("OPENAI_API_BASE"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    default_model: str = field(default_factory=lambda: _env("MODEL"))
    vl_model: str = field(default_factory=lambda: _env("VL_MODEL"))
    item_model: str = field(default_factory=lambda: _env("ITEM_MODEL"))

    # ====================== Embedding ======================
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    embedding_api_base: str = field(
        default_factory=lambda: _env("EMBEDDING_API_BASE")
    )
    embedding_api_key: str = field(
        default_factory=lambda: _env("EMBEDDING_API_KEY") or _env("OPENAI_API_KEY")
    )
    embedding_dim: int = field(
        default_factory=lambda: _env_int("EMBEDDING_DIM", 1024)
    )

    # ====================== MinerU PDF 解析 ======================
    mineru_api_base: str = field(default_factory=lambda: _env("MINERU_API_BASE"))
    mineru_api_key: str = field(default_factory=lambda: _env("MINERU_API_KEY"))
    mineru_model_version: str = field(
        default_factory=lambda: _env("MINERU_MODEL_VERSION", "vlm")
    )

    # ====================== Milvus 向量库 ======================
    milvus_url: str = field(default_factory=lambda: _env("MILVUS_URL"))
    milvus_token: str = field(default_factory=lambda: _env("MILVUS_TOKEN"))
    chunks_collection: str = field(default_factory=lambda: _env("CHUNKS_COLLECTION"))
    item_name_collection: str = field(
        default_factory=lambda: _env("ITEM_NAME_COLLECTION")
    )
    entity_name_collection: str = field(
        default_factory=lambda: _env("ENTITY_NAME_COLLECTION")
    )

    # ====================== MinIO 对象存储 ======================
    minio_endpoint: str = field(default_factory=lambda: _env("MINIO_ENDPOINT"))
    minio_access_key: str = field(default_factory=lambda: _env("MINIO_ACCESS_KEY"))
    minio_secret_key: str = field(default_factory=lambda: _env("MINIO_SECRET_KEY"))
    minio_bucket: str = field(default_factory=lambda: _env("MINIO_BUCKET_NAME"))
    minio_secure: bool = field(
        default_factory=lambda: _env_bool("MINIO_SECURE", False)
    )

    # ====================== 路径 ======================
    project_root: str = field(default_factory=lambda: _env("PROJECT_ROOT", "."))
    tmp_dir: str = field(default_factory=lambda: _env("TMP_DIR", "./tmp"))
    upload_dir: str = field(default_factory=lambda: _env("UPLOAD_DIR", "./uploads"))

    # ====================== 日志 ======================
    log_level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO").upper()
    )

    # ====================== 实例方法 ======================
    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从当前进程环境变量构建 ImportConfig 实例"""
        return cls()

    @property
    def minio_full_url(self) -> str:
        """拼接 MinIO 完整访问 URL"""
        protocol = "https://" if self.minio_secure else "http://"
        return f"{protocol}{self.minio_endpoint}"

    def validate(self) -> None:
        """校验关键配置；缺失则抛 ConfigurationError（在节点首次执行时调用）"""
        from .exceptions import ConfigurationError
        missing = {
            "OPENAI_API_KEY": self.openai_api_key,
            "MILVUS_URL": self.milvus_url,
            "MINIO_ENDPOINT": self.minio_endpoint,
        }
        not_set = [k for k, v in missing.items() if not v]
        if not_set:
            raise ConfigurationError(
                message=f"关键环境变量缺失: {', '.join(not_set)}",
                node_name="config",
            )


# ====================== 全局单例 ======================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """
    获取全局配置单例；首次调用时从环境变量构建。
    BaseNode 中已通过 `self.config = config or get_config()` 自动注入。
    """
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config


def reload_config() -> ImportConfig:
    """
    重新读取 .env 并重建单例（覆盖已有环境变量），常用于测试或运行时切换配置。
    """
    global _config
    load_dotenv(override=True)
    _config = ImportConfig.from_env()
    return _config


def reset_config() -> None:
    """清空单例；下一次 get_config() 时将重新构建。"""
    global _config
    _config = None
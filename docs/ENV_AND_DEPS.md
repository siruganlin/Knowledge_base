# 环境与依赖说明

本文对应当前项目的 `config/`、`utils/` 和 Web 服务。历史课程参考仓库为 `gitee.com/lzz66666/Knowledge_Project`，配置名称以当前源码和根目录 `.env.example` 为准。

## 环境要求

- Python ≥ 3.11；BGE-M3 建议使用 Python 3.11 或 3.12 的独立环境。
- Milvus 2.4.x、MinIO、MongoDB 服务，连接地址由使用者配置。
- 本地 BGE-M3 模型文件；有 GPU 时可切换模型设备。
- 模型服务、MinerU、DashScope 重排序和百炼联网 MCP 的访问权限。

## 安装

```bash
python -m venv .venv
# 激活虚拟环境后执行：
python -m pip install -r requirements.txt
```

`requirements.txt` 使用 `pymilvus[model]>=2.4.15,<2.5` 对应 Milvus 2.4.x，包含 `pymilvus.model` 模型扩展，并显式安装 BGE-M3 所需的 `FlagEmbedding==1.3.3`。如果更换 Milvus 服务端版本，请同时核对客户端版本。

文件上传接口依赖 `python-multipart`，已在依赖清单中列出。BGE-M3 会引入 PyTorch 等较大的依赖；GPU 环境需要按机器情况安装对应的 PyTorch。

## 配置

将项目根目录的 `.env.example` 复制为 `.env`，并从项目根目录启动服务。

| 变量 | 用途 |
|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` | OpenAI 兼容模型接口地址和凭据 |
| `LLM_DEFAULT_MODEL` / `LLM_DEFAULT_TEMPERATURE` | 默认模型和温度 |
| `VL_MODEL` / `ITEM_MODEL` | 图像理解和商品名识别模型 |
| `BGE_M3_PATH` | 本机 BGE-M3 模型目录 |
| `BGE_M3` | 模型标识，用于日志 |
| `BGE_DEVICE` / `BGE_FP16` | 模型设备和半精度开关；CPU 建议 `cpu` / `0` |
| `MILVUS_URL` | Milvus 地址，例如 `http://127.0.0.1:19530` |
| `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` | 文档片段和商品名集合 |
| `MINIO_ENDPOINT` | MinIO 地址，例如 `127.0.0.1:9000`，不包含协议 |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 自行配置的 MinIO 凭据 |
| `MINIO_BUCKET_NAME` / `MINIO_IMG_DIR` | 存储桶和图片对象前缀 |
| `MONGO_URL` / `MONGO_DB_NAME` | MongoDB 地址和历史会话数据库 |
| `OPENAI_API_KEY` | 当前代码中 DashScope 重排序和百炼 MCP 使用的 Key |
| `TEXT_RERANK_MODEL` / `TEXT_RERANK_INSTRUCT` | 重排序模型和指令 |
| `MCP_DASHSCOPE_BASE_URL` | 已开通的百炼联网 MCP 服务地址 |
| `MINERU_BASE_URL` / `MINERU_API_TOKEN` | MinerU 在线解析接口及 Token |
| `DATA_BASED_ROOT_DIR` | 上传和解析文件的本地目录，例如 `./data` |

`LLM_API_KEY` 和 `OPENAI_API_KEY` 可以使用同一个具备对应服务权限的 DashScope Key。服务示例地址只适用于本机部署；远程服务应改为自己的主机地址。

`processor/import_processor/import_config.py` 还保留了通用配置字段；当前外部服务调用主要读取上表对应的 `config/` 配置。不要将旧课程文档中的变量名直接替代上表名称。

## 启动

```bash
# 分别在两个终端中执行：
python -m web.api.query_service
python -m web.api.import_service
```

查询页面为 `http://127.0.0.1:8001/chat.html`，入库页面为 `http://127.0.0.1:8002/import.html`。运行前请确保服务可连接，并完成 `.env` 和模型路径配置。

## 发布说明

`.env`、本地上传文件和解析产物已加入忽略规则。连接凭据、SSH 登录信息应保存在本地或专门的凭据管理服务中。

本项目旧 Git 历史曾包含登录凭据；公开当前源码时请使用不带 `.git` 历史的干净文件重新建仓，并更换已使用过的旧凭据。

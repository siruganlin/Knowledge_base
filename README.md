# 智能知识库检索系统（Knowledge_base_0525）

基于 **LangGraph + Milvus + BGE-M3** 的垂直领域知识库检索与问答系统。面向技术手册、规程标准等专业文档，实现从 PDF 解析入库到带引用答案输出的完整链路。

> 本项目在开源课程项目基础上二次开发，重点重构了文档切分、混合召回与重排序策略。

---

## 功能特性

- **文档智能入库**：PDF → 版面解析 → 图片抽取 → 语义切分 → 向量化 → 写入 Milvus
- **双路混合召回**：稠密语义召回（BGE-M3）+ 稀疏关键词召回，经 RRF 倒数排名融合
- **重排序精排**：交叉编码重排模型对候选片段二次打分，提升 Top-K 精度
- **带引用问答**：答案强制标注来源片段，可回溯到原文条款
- **对话历史**：MongoDB 持久化会话上下文，支持多轮追问
- **流式输出**：SSE 推流，降低首字延迟

---

## 技术栈

| 层次 | 技术 |
|---|---|
| 工作流编排 | LangGraph 1.2 / LangChain |
| 向量模型 | BGE-M3（`flagembedding`） |
| 向量数据库 | Milvus 2.4（standalone） |
| 对象存储 | MinIO |
| 文档数据库 | MongoDB |
| 文档解析 | MinerU（版面还原 + 图片抽取） |
| 重排序 | 通义千问 Rerank（HTTP 调用） |
| Web 服务 | FastAPI + Uvicorn |

---

## 系统架构

```
                    ┌──────────────── import_processor ─────────────────┐
   PDF 文档 ───────►│ 解析 → 图片抽取 → 语义切分 → 向量化 → 入库        │──► Milvus / MinIO / MongoDB
                    └───────────────────────────────────────────────────┘

                    ┌──────────────── query_porcessor ──────────────────┐
   用户提问 ───────►│ 清洗改写 → 双路召回 → RRF 融合 → 重排 → 答案生成  │──► 带引用的答案
                    └───────────────────────────────────────────────────┘
```

---

## 目录结构

```
Knowledge_base_0525/
├── config/                    # 服务配置（LLM / Embedding / Milvus / MinIO / Rerank / MCP）
├── processor/
│   ├── import_processor/      # 入库流程
│   │   ├── main_graph.py      #   LangGraph 节点编排
│   │   └── nodes/             #   a_入口 / b_PDF转MD / c_图片处理 / d_文档切分
│   │                          #   e_商品名识别 / g_写入 Milvus
│   └── query_porcessor/       # 查询流程
│       ├── main_graph.py
│       └── nodes/             #   a_名称对齐 / b_向量召回 / c_HyDE召回 / d_MCP联网
│                              #   e_RRF融合 / f_重排 / g_答案输出
├── utils/                     # Milvus / MinIO / Mongo / Embedding / Rerank / SSE 工具
├── web/
│   ├── api/                   # FastAPI 服务（入库 8002 / 查询 8001）
│   └── page/                  # 前端页面
├── tool/logger.py             # 日志
├── docs/ENV_AND_DEPS.md       # 环境与依赖说明
└── requirements.txt
```

---

## 快速开始

### 1. 环境要求

- Python ≥ 3.11
- Milvus 2.4.x、MinIO、MongoDB 各一个实例（建议 Docker 部署）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

复制 [`.env.example`](.env.example) 为 `.env`，逐项填写模型、Milvus、MinIO、MongoDB、MinerU 和 MCP 配置，并将 `BGE_M3_PATH` 改为本机模型目录。

`LLM_API_KEY` 用于模型调用，`OPENAI_API_KEY` 用于 DashScope 重排序和百炼 MCP；可以填写同一个具备对应权限的 DashScope Key。完整变量说明见 [`docs/ENV_AND_DEPS.md`](docs/ENV_AND_DEPS.md)。

### 4. 启动服务

```bash
# 查询服务（8001）
python -m web.api.query_service

# 入库服务（8002）
python -m web.api.import_service
```

---

## 使用流程

1. 调入库接口上传 PDF，等待 `import_processor` 执行完成（各节点进度见日志）
2. 确认 Milvus 中已有对应 collection 数据
3. 调查询接口提问，接收 SSE 流式答案

---

## 已知限制

- 图片引用路径依赖入库时的对象存储地址，迁移环境后需重建索引
- 重复导入同一 PDF 会产生重复片段，幂等去重尚未实现
- 超长章节跨段的上下文截断仍有优化空间

---

## License

[MIT](LICENSE)

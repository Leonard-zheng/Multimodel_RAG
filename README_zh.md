MultiRAG：多模态检索增强生成（RAG）

概览
- 目标：解析 PDF（文本/表格/图片），对内容进行摘要，写入向量库，并基于文档进行问答。
- 技术栈：LangChain、Google GenAI（LLM + Embedding）、Chroma、Unstructured、python-dotenv。

功能
- PDF 分区：提取文本、表格和 base64 图片。
- 摘要缓存：文本与图片摘要均使用可复用的 LLM 实例并缓存结果。
- 多向量检索：Chroma 向量库持久化存储。
- 简洁 RAG：构建上下文 → LLM 回答（可扩展为返回引用来源）。
- 统一日志与错误处理。

快速开始
- 推荐 Python 3.10+
- 创建虚拟环境：`python -m venv .venv && source .venv/bin/activate`
- 安装依赖：`pip install -r requirements.txt`
- 配置环境变量：`cp .env.example .env` 并设置 `GOOGLE_API_KEY`
- 运行：`python main.py`

配置
- 核心设置在 `src/config.py`（dataclass 默认值）。`.env` 主要用于提供第三方 SDK 的密钥（如 `GOOGLE_API_KEY`）。
- 提示词模板：`config/prompt.yml`（文本/表格与图片提示词）。
- 默认输入 PDF：`content/attention-is-all-you-need.pdf`。

存储与持久化
- 向量库：`./chroma_db/`。
- 摘要缓存：`./cache/summaries.json`。
- 原始内容 docstore：`./docstore.pkl`（pickle）。

使用说明
- 首次运行会解析默认 PDF、生成摘要并建立索引。
- 后续运行会复用已有索引与缓存。
- 可在 `main.py` 中修改 `query`，或改造成你自己的 CLI/交互方式。


常见问题
- 缺少 `GOOGLE_API_KEY`：请在 `.env` 中设置。
- 重新构建索引：删除 `./chroma_db/` 以强制重新向量化。
- docstore 格式：程序读取 `docstore.pkl`


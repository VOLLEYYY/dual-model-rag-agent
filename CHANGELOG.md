# 代码更新日志（CHANGELOG）

> 用途：记录每次代码变更，便于**回滚、定位问题、面试复盘**。
> 规范：按版本倒序，每次变更写「改了什么 → 为什么 → 怎么验证 → 可能踩的坑」。

---

## [0.5.6] — 2026-08-30 — 性能优化（模型B 换非推理模型）+ UI 流式化 + 知识库扩充

### 一句话

三件事：模型B 从推理模型换成普通对话模型（校验提速 17~38 倍）、Gradio UI 从非流式切到 SSE 流式（首字延迟降到 ~0.8s）、知识库新增 6 篇 RAG / Agent 文档。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `.env` / `.env.example` / `app/config.py` | `MODEL_B_NAME` 由 `deepseek-v4-pro` → `deepseek-chat`（含默认值与模板注释同步） | 模型B 换非推理模型，消除评审的长时间「思考」 |
| `app/ui/chat.py` | 新增 `ask_stream()` 异步 generator + `_format_details()`；两个提交事件的 `fn` 改为 `ask_stream`；保留 `ask()` 作非流式备选 | 前端走 `/api/ask/stream`，逐 token 显示回答，大幅降低感知延迟 |
| `data/knowledge_base/` | 新增 6 篇文档：`rag_hybrid_retrieval.md` / `rag_chunking_strategies.md` / `rag_evaluation.md`（RAG）、`agent_react.md` / `agent_tools.md` / `agent_memory_multiagent.md`（Agent） | 扩充知识库，覆盖检索进阶与 Agent 基础 |

### 关键设计（面试可讲）

1. **模型B 换非推理模型**：评审打分不需要推理能力。推理模型（`deepseek-v4-pro`）即使 prompt 要求「只输出 JSON」，仍会先「思考」再落笔，这部分推理耗时无法通过 prompt 压掉。换 `deepseek-chat` 后校验从 15~35s 降到 ~0.9s，评分结论不变（9~10 分）。
2. **流式感知延迟**：非流式要等全流程（检索 + 生成 + 校验）跑完才一次性返回；流式把「等全流程」变成「等首 token」，模型B 校验在用户读答案时后台进行。稳态单次请求从 17~37s 降到 ~2.8s，首 token ~0.8s。

### 验证（实测耗时对比）

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 模型B 校验 | 15.3 ~ 34.7 s | 0.75 ~ 1.17 s（平均 0.90 s） |
| 稳态首 token（TTFT） | —（非流式，无首字概念） | 838 ms |
| 稳态单次请求总耗时 | 17.14 ~ 36.95 s | 2.80 s |

评审质量：`overall` 稳定 9~10 分，`need_revision` 结论一致（均为 False）。

> 冷启动提醒：后端进程首次请求需加载 embedding 模型（bge-small-zh-v1.5）约 13.5s，
> 这是每个进程只发生一次的成本，之后检索器单例缓存，稳态只剩 ~0.8s 首 token。

---

## [0.5.5] — 2026-08-29 — GitHub 上线前门面收尾（版本号 / 进度 / 描述对齐）

### 一句话

上线前把「门面」做一致：版本号对齐到 `0.5.4`、README 进度与路线表更新到 Day 18/19/21、`main.py` 描述修正过时措辞。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/config.py` | `app_version` 0.1.0 → 0.5.4 | 与 CHANGELOG 版本对齐，`/api/health` 返回真实版本 |
| `.env.example` | `APP_VERSION=0.5.4` | 与 config 对齐 |
| `README.md` | 顶部「当前进度」更新；「七、后续路线」补 Day 18/19/21；health 返回示例版本改为 0.5.4 | 文档与代码进度一致 |
| `app/main.py` | `description` 修正（去掉「Qwen2.5 生成」过时措辞，改为「模型A 生成 + 模型B 独立评审，可切换 DeepSeek/Ollama」） | OpenAPI 文档描述准确 |

### 验证

```bash
python -c "from app.config import settings; print(settings.app_version)"   # → 0.5.4
pytest -v   # 6 passed（版本号/描述为字符串改动，不影响逻辑）
```

---

## [0.5.4] — 2026-08-29 — README 补 Docker 一键部署 + Mermaid 架构图（Day 21）

### 一句话

README 补「Docker 一键部署」章节（`docker compose up -d` + 三件套 curl 验证 + 容器内地址差异表），并加 Mermaid 架构图；同时同步测试引用（`python test_*.py` → `pytest -v`）。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `README.md` | 新增 2.5「Docker 一键部署」章节 | 别人 clone 后能一键起整套栈并验证 |
| `README.md` | 「一、项目是什么」加 Mermaid 架构图 | 一张图讲清 提问→检索→生成→安检→校验→返回 全链路 |
| `README.md` | 测试引用同步：2.4/3.1/4.9/六节 的 `python test_*.py` → `pytest -v`，目录结构补 `tests/` 与 `pytest.ini` | 与 [0.5.3] 测试迁移保持一致 |

### 关键设计（面试可讲）

1. **Docker 章节讲清「容器内地址差异」**：宿主机跑用 `127.0.0.1`，容器内跨服务用服务名（`qdrant`/`ollama`）或 `host.docker.internal`（连宿主 DeepSeek 代理）。这是「本地能跑 ≠ 容器能跑」的典型坑，一张表讲明白。
2. **Mermaid 图即「口述版」架构**：面试脱稿讲项目时，这张图就是骨架——先检索、再生成、过安检、独立模型评审、必要时修订，五步闭环。

### 验证

```bash
# compose 配置（历史已验证，见 [0.4.0]）
docker compose config   # 无 error

# 三件套（Docker Desktop 启动 + 容器 Up 后）
curl http://localhost:8000/api/health    # → {"status":"ok",...}
curl http://localhost:6333/healthz       # → 200（1.19 用 /healthz，/health 返回 404）
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" -d '{"question":"什么是 RAG？"}'
```

### 可能踩的坑

| 现象 | 原因 / 解决 |
|------|------------|
| `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` | Docker Desktop 未启动；README 2.5 已补前提说明 |

---

## [0.5.3] — 2026-08-29 — 测试脚本升级为 pytest 套件（Day 19）

### 一句话

把根目录的 `test_rag.py` / `test_dual_model.py` 改造成 pytest 用例，迁入 `tests/` 目录，`pytest -v` 一键回归，从「脚本」升级到「工程化测试」。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `tests/conftest.py`（新增） | sys.path 注入 + UTF-8 输出 + `project_root`/`knowledge_base_path` 两个 session fixture | 统一路径与编码，任意 cwd 下跑都不出错 |
| `tests/test_rag.py`（新增） | 4 个用例：TF-IDF 检索 / 稠密检索 / 安全检查 / 单模型流水线 | 覆盖 RAG + 安全 + 单模型链路 |
| `tests/test_dual_model.py`（新增） | 2 个用例：评审 JSON 解析 / 双模型流水线 5 类问题 | 覆盖双模型校验链路 |
| `pytest.ini`（新增） | `testpaths=tests` + `pythonpath=.` | 指定测试目录 + 保证 `import app` 可用 |
| `test_rag.py` / `test_dual_model.py` | 删除 | 已迁入 `tests/` |

### 关键设计（面试可讲）

1. **离线 / 在线分层**：离线用例（TF-IDF、安全检查、评审解析）确定性必过；需外部依赖的用例（稠密检索、LLM 流水线）在依赖/服务缺失时 `pytest.skip`，不拖累 CI——这是「测试也要能降级」。
2. **不用 pytest-asyncio**：async 逻辑用 `asyncio.run` 包在同步 `test_*` 函数内，少一个依赖、跨环境更稳，也避免插件版本兼容问题。
3. **conftest 统一 fixture**：`knowledge_base_path` 用 `PROJECT_ROOT` 拼绝对路径，替换原脚本的 `./data/knowledge_base` 相对路径，测试不再依赖「必须从 HELLO 目录运行」。

### 验证

```bash
cd HELLO
pytest -v
# → 6 passed（TF-IDF 检索 / 稠密检索 / 安全检查 / 单模型流水线 / 评审解析 / 双模型流水线）✅
```

### 可能踩的坑

| 现象 | 原因 / 解决 |
|------|------------|
| `UnicodeDecodeError: 'gbk' codec can't decode ...` | pytest.ini 含中文注释时，iniconfig 用系统默认 GBK 编码读 .ini 失败；`pytest.ini` 保持纯 ASCII |

---

## [0.5.2] — 2026-08-29 — 模型调用容错：超时 + 重试 + 指数退避（Day 18）

### 一句话

给 LLM 调用补上「超时 + 重试 + 降级」容错三件套：瞬时故障自动重试（指数退避），重试耗尽后优雅降级不崩。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/core/llm.py` | `invoke` 加 `@retry`（tenacity，最多 3 次、指数退避 1→10s）；新增 `_is_retryable_error`；`AsyncOpenAI` 设 `max_retries=0` | 重试由 tenacity 统一控制，只重试瞬时错误 |
| `app/core/agent_pipeline.py` | `run_pipeline` 的模型A 生成 / 模型B 校验各包一层 `asyncio.wait_for(timeout=settings.llm_timeout)` | 流水线级整体超时兜底，防止无限等待 |
| `requirements.txt` | 新增 `tenacity>=8.2.0` | 重试依赖 |

### 关键设计（面试可讲）

1. **三层容错**：单次调用超时（`AsyncOpenAI` 的 `timeout`）→ tenacity 指数退避重试（最多 3 次）→ `asyncio.wait_for` 整体兜底 → `try/except` 降级。层层递进，任何一层兜住都不会让请求崩掉。
2. **只重试瞬时错误**：`_is_retryable_error` 判定——连接失败 / 超时 / 限流（429）/ 5xx 服务端错误可重试；4xx（参数错、鉴权失败）不重试，因为重试也白费。这是「重试要分错误类型」的工程意识。
3. **重试只加在一次性生成，不加流式**：`stream`（SSE）不重试——流式已向用户吐出部分 token，重试会重复输出、破坏体验。流式失败直接降级（返回检索片段）。
4. **`max_retries=0` 关闭 SDK 内部重试**：openai SDK 默认 `max_retries=2`，与 tenacity 叠加会变成 3×3 次尝试、耗时不可控（实测连接失败要等 27s）。关闭后重试次数明确 = 3 次，耗时降到约 10s。
5. **`reraise=True` 重试耗尽抛原始异常**：让上层 `except Exception` 拿到的是 `APIConnectionError` 而非 tenacity 的 `RetryError`，降级日志能准确显示失败原因。

### 验证

```python
# 1. 重试判定：5xx 可重试、4xx 不可重试
_is_retryable_error(_FakeStatus(502))  # True
_is_retryable_error(_FakeStatus(400))  # False

# 2. 连接失败（base_url 指向不存在端口）→ 重试 3 次 → reraise 抛异常
LLMClient(base_url="http://127.0.0.1:59999/v1").invoke(...)
# → 抛 APIConnectionError，耗时 ~10s（3 次尝试 + 退避 1s/2s）✅

# 3. wait_for 超时抛 TimeoutError，被 run_pipeline 的 except Exception 捕获 → 降级返回检索片段
```

> 完整端到端：把 `.env` 的 `MODEL_A_BASE_URL` 指向不存在端口，`POST /api/ask` 仍返回
> `success=true`，回答为「⚠️ 模型生成失败（APIConnectionError）…检索片段」，流水线不崩。

### 可能踩的坑

| 现象 | 原因 / 解决 |
|------|------------|
| 重试耗时偏长（27s） | openai SDK 默认 `max_retries=2` 与 tenacity 叠加；已在 `AsyncOpenAI(..., max_retries=0)` 关闭，重试统一走 tenacity |
| 重试后拿到 `RetryError` | tenacity 默认包一层；设 `reraise=True` 抛原始异常，降级日志更直观 |

---

## [0.5.1] — 2026-08-28 — 知识库重建索引端点 + 前端按钮

### 一句话

新增 `POST /api/knowledge/reindex` 端点 + Gradio「重建知识库索引」按钮：往 `data/knowledge_base/` 加文档后，一键重新分块/向量化/写入 Qdrant，不必手动删 `qdrant_data` 或重启。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/core/rag.py` | `QdrantRetriever` 新增 `chunk_count` 属性 | 与 `TFIDFRetriever.chunk_count` 对称，供 reindex 返回块数 |
| `app/core/agent_pipeline.py` | 新增 `reindex_knowledge_base()`：列文档清单 + `get_retriever(force_reload=True)` 重建 | 绕过幂等摄入，强制重新索引 |
| `app/models/schemas.py` | 新增 `ReindexResponse` | reindex 接口的响应契约 |
| `app/api/routes.py` | 新增 `POST /api/knowledge/reindex` | 对外暴露重建索引能力 |
| `app/ui/chat.py` | 新增「知识库管理」折叠区 + 重建按钮 | 前端一键触发，展示文档清单与块数 |

### 关键设计（面试可讲）

1. **force 重建绕开幂等**：`get_retriever(force_reload=True)` 内部 `ingest_directory(force=True)` 会先删 collection 再重建，解决「加文档后普通重启不生效」的缓存问题。
2. **接口返回文档清单 + 块数**：reindex 返回「模式/文档数/块数/文档清单」，前端点一下就能看到重建结果，比「删目录重启」透明得多。
3. **chunk_count 对称属性**：稠密/稀疏检索器统一暴露 `chunk_count`，reindex 不关心底层实现。

### 验证

```bash
curl -X POST http://127.0.0.1:8002/api/knowledge/reindex
# → {"success":true,"mode":"dense","document_count":2,"chunk_count":4,
#    "documents":["fastapi_guide.md","rag_intro.md"],"message":"索引重建完成..."} ✅
```

---

## [0.5.0] — 2026-08-28 — Langfuse 可观测性 + Gradio 聊天界面（Day 14/16）

### 一句话

接 Langfuse 可观测性（`@observe()` 装饰 + trace 打点，未配置密钥时自动禁用）+ 新增 Gradio 聊天界面（调 `/api/ask`，展示回答 + 模型B 评审详情）。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/core/observability.py`（新增） | `LangfuseObserver` 门面：`observe()` 装饰器、`update_trace()`/`update_span()`/`flush()`；未装依赖/未配密钥时全部 no-op | 可观测性可选依赖 + 优雅降级，不侵入 rag/llm 底层 |
| `app/core/agent_pipeline.py` | `run_pipeline` 加 `@observer.observe` + 分步 `update_trace`（检索模式/命中数/评分/来源/安检结果） | 每个请求在 Langfuse 生成一条 trace，含输入/输出/元数据 |
| `app/config.py` | 新增 `langfuse_enabled` / `langfuse_public_key` / `langfuse_secret_key` / `langfuse_base_url` | 与 embedding/model provider 对称，配置化开关 |
| `app/ui/chat.py` + `app/ui/__init__.py`（新增） | Gradio 界面：问题输入 + 双模型开关 + top_k 滑块 + 回答/评审详情输出，`httpx` 调后端 `/api/ask` | 展示层，面试演示用 |
| `requirements.txt` | 打开 `langfuse>=3.0,<4`、`gradio>=4.0` | 正式启用两个依赖 |
| `.env` / `.env.example` | 新增 `LANGFUSE_ENABLED` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | 配置对齐 |

### 关键设计（面试可讲）

1. **可观测性 = 可选依赖 + 优雅降级**：`observability.py` 用「延迟导入 + 密钥探活」，未装 langfuse 或未配密钥时所有打点自动 no-op，主流程零开销、零影响——和 RAG「接口不变、失败降级」是同一套工程思想。
2. **只在编排层打点**：Langfuse 装饰器只加在 `run_pipeline`（编排层），不侵入 `rag.py`/`llm.py` 底层，保持「每层只依赖下层」的分层约束。
3. **Langfuse v3 API 选型**：v4 移除了 `langfuse.decorators` 与 `update_current_trace`（metadata 也限 str），故锁定 `langfuse>=3.0,<4`，用 `observe()` + `get_client().update_current_trace()`——API 稳定、metadata 支持任意 JSON。
4. **trace 记录什么**：input（问题/是否双模型/top_k）、output（最终回答/评分/是否修订）、metadata（检索模式 dense/sparse、命中数、来源）——面板上能直接看到「这轮用了稠密还是稀疏检索、模型B 打几分、是否修订」。

### 验证

```bash
# 无密钥降级（不破坏主流程）
python -c "from app.core.agent_pipeline import observer; print(observer.is_enabled())"   # → False

# 主流程端到端（新代码，8001 端口）
python -m uvicorn app.main:app --port 8001 &
curl -X POST http://127.0.0.1:8001/api/ask -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?","top_k":3}'
# → success=true, review_score=7.0, need_revision=true, sources=[rag_intro.md, fastapi_guide.md] ✅

# Gradio 启动
python -m app.ui.chat   # → http://localhost:7860 返回 200 ✅
```

> Langfuse 面板出 Trace 的最终验证：在 `.env` 填入 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
> （日本地区 `LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com`）后重启服务，`/api/ask` 一次即产生 trace。

### 可能踩的坑

| 现象 | 原因 / 解决 |
|------|------------|
| `ModuleNotFoundError: No module named 'langfuse.decorators'` | 装了 langfuse v4（4.x 移除 decorators 模块）；本项目按 v3 编写，`pip install "langfuse>=3.0,<4"` 锁定 |
| Langfuse 面板看不到 trace | ① `.env` 未填 `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`；② `LANGFUSE_BASE_URL` 与注册地区不符（日本地区必须 `https://jp.cloud.langfuse.com`）；③ 启动日志有「未配置密钥」提示 |
| `Langfuse client initialized without public_key` 警告 | 无密钥时 client 被禁用，属预期（主流程不受影响）；填密钥后消失 |
| 中文日志/输出乱码 | Windows 控制台 GBK 编码问题，不影响功能 |

---

## [0.4.0] — 2026-08-27 — 模型后端可切换（MODEL_PROVIDER）+ Docker 化收尾

### 一句话

给模型A/B 加 `MODEL_PROVIDER` 开关（`deepseek`/`ollama`），`.env` 一行切换后端；修正 `docker-compose.yml` 的硬编码与 Qdrant 连接 bug，取消 Ollama 注释，新增 `.dockerignore` 防止密钥进镜像。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/config.py` | 新增 `model_provider` / `ollama_base_url` / `ollama_model_a` / `ollama_model_b`；新增 `resolve_model_backend(role)` | 与 `EMBEDDING_PROVIDER` 对称，模型后端可配置 |
| `app/core/agent_pipeline.py` | `ModelAGenerator` / `ModelBReviewer` 的 `__init__` 缺省时改走 `resolve_model_backend("a"/"b")` | 模型构建按 provider 解析，显式传参仍保留 |
| `.env` / `.env.example` | 新增 `MODEL_PROVIDER` / `OLLAMA_*` | 配置化后端切换 |
| `docker-compose.yml` | 去硬编码 `MODEL_*_BASE_URL`，改 `env_file` 透传 + `QDRANT_URL=http://qdrant:6333` + `host.docker.internal`；取消 Ollama 注释；`depends_on` 加 ollama；移除废弃的 `version` | 修「容器内退回本地嵌入式 Qdrant」的 bug，一键起整套栈 |
| `Dockerfile` | `apt` 换清华 Debian 源；`pip` 换清华 PyPI 源 | 容器内 apt/pip 访问官方源极慢/502，换国内源加速构建 |
| `.dockerignore`（新增） | 忽略 `.env`/`qdrant_data`/`data`/缓存等 | 防止 `COPY . .` 把密钥和数据打进镜像 |

### 关键设计（面试可讲）

1. **后端 provider 开关**：模型A/B 和 embedding 一样，用 `MODEL_PROVIDER`（deepseek/ollama）切换，体现「接口不变、只换实现」，切换后端不动业务代码。
2. **容器 vs 宿主的地址差异**：宿主机跑用 `127.0.0.1`；容器内跨服务用服务名（`qdrant`/`ollama`）或 `host.docker.internal`（连宿主）。compose 用 `environment` 覆盖、`env_file` 透传，两者分离。
3. **`.dockerignore` 防密钥泄露**：`COPY . .` 会把 `.env` 打进镜像，`.dockerignore` 是容器化的安全底线。

### 验证

```bash
# 后端解析（默认 deepseek）
python -c "from app.config import settings; print(settings.model_provider, settings.resolve_model_backend('a')[0], settings.resolve_model_backend('a')[2])"
# → deepseek http://127.0.0.1:20128/v1 ds/deepseek-chat

# compose 语法
docker compose config
# → 无 error（仅提示 version 已废弃，已移除）
```

容器化端到端实测（2026-08-28）：

```bash
docker compose build agent-api              # 换国内源后构建成功
docker compose up -d qdrant                 # 起 Qdrant
docker compose up -d --no-deps agent-api    # 起 FastAPI（跳过 Ollama）
curl http://localhost:8000/api/health       # → {"status":"ok",...}
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" -d '{"question":"What is RAG?"}'
# → success=true，检索命中 rag_intro.md，模型A 生成 + 模型B 评审(8.0)触发修订，双模型制衡在容器内全链路跑通 ✅
```

### 可能踩的坑

| 现象 | 原因 / 解决 |
|------|------------|
| 容器内问答连不上 DeepSeek 代理 | 容器内 `127.0.0.1` 指向容器自己；compose 已用 `host.docker.internal:20128` 覆盖（需宿主 20128 端口监听非仅回环） |
| 容器内稠密检索没连 qdrant 容器 | 原 compose 传了 `QDRANT_HOST/PORT`（无效旧字段），已改 `QDRANT_URL=http://qdrant:6333` |
| `version` 属性 warning | Compose v2 已废弃，移除即可 |
| 构建时 `apt-get` 502 / 极慢(51kB/s) | 容器内访问 `deb.debian.org` 走 VPN 出口；Dockerfile 已 sed 换清华 Debian 源 |
| 镜像 8.64GB 偏大 | `torch` 默认装 CUDA 版（一堆 nvidia 包）；后续可改 CPU 版 torch（`--index-url https://download.pytorch.org/whl/cpu`）精简到 ~2GB |
| qdrant 容器 `unhealthy` | 镜像不含 `curl`，且 1.19 健康端点是 `/healthz`（`/health` 返回 404）；healthcheck 已改 bash `/dev/tcp` 探测端口 |

---

## [0.3.1] — 2026-08-26 — 固化 HF_ENDPOINT 镜像到配置

### 一句话

把 HF_ENDPOINT 镜像固化到 `.env` + `config.py`，启动服务无需再手动 `export`，避免加载 bge 模型卡在连官方 huggingface.co。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/config.py` | 新增 `hf_endpoint` 字段；`settings` 初始化后 `os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)` | 未显式设置时自动注入镜像；在延迟导入 sentence_transformers 前生效 |
| `.env` / `.env.example` | 新增 `HF_ENDPOINT=https://hf-mirror.com` | 与 config 对齐 |

### 验证

不带 `HF_ENDPOINT` 前缀启动服务，SSE 请求 `retrieving` 立即流出、不再卡住 ✅

---

## [0.3.0] — 2026-08-26 — SSE 流式端点（Day 13）

### 一句话

补 `POST /api/ask/stream` 流式端点：检索→生成→安检→校验逐步以 SSE 事件推送，生成环节逐 token 吐出，降低首字延迟（TTFT）。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/core/agent_pipeline.py` | 新增 `run_pipeline_stream()` 异步生成器，逐步 yield 事件 dict | 复用现有组件拆成流式；降级策略与 `run_pipeline` 一致 |
| `app/api/routes.py` | 新增 `POST /ask/stream`，`StreamingResponse` + `json.dumps(ensure_ascii=False)` | 把事件 dict 序列化为 SSE 格式 |

### 事件序列

`retrieving → retrieved → generating → token*(逐 token) → safety → review → done`

### 如何验证

```bash
# 启动服务（务必带 HF_ENDPOINT，否则加载 bge 模型会卡在连官方 HF）
HF_ENDPOINT=https://hf-mirror.com python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 流式实测（-N 禁用缓冲）
curl -N -X POST http://127.0.0.1:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是 RAG？","top_k":3}'
```

- `use_dual_model=false`：看到逐 token 吐出 + done
- `use_dual_model=true`（默认）：review 事件带 `score=8.0, need_revision=true`，模型B 发现「英文提问中文回答」并修订为英文

### 可能踩的坑 / 排查

| 现象 | 原因 / 解决 |
|------|------------|
| SSE 首事件迟迟不出、服务卡住 | 服务进程未设 `HF_ENDPOINT`，加载 bge 模型时连官方 huggingface.co 超时重试；启动时加 `HF_ENDPOINT=https://hf-mirror.com` |
| `There was an error parsing the body` | Git Bash 里 curl `-d` 传中文的 GBK/UTF-8 编码问题；用英文问题或 `-d @file.json`（UTF-8 文件） |

---

## [0.2.1] — 2026-08-26 — 跑通稠密检索 + 适配 qdrant-client 新版 API

### 一句话

补齐 sentence-transformers 依赖并**实际跑通稠密检索**（`mode: dense`）；修复 qdrant-client >=1.10 移除 `search()` 导致的 API 兼容问题。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/core/rag.py` | `QdrantRetriever.search()`：`client.search(query_vector=)` → `client.query_points(query=).points` | 适配 qdrant-client 1.19.0（>=1.10 已移除 `search`） |

### 依赖变化（本次实装）

| 包 | 版本 | 说明 |
|----|------|------|
| `sentence-transformers` | 6.0.0 | 新增，本地 embedding 推理 |
| `transformers` | 4.57.6 → 5.15.1 | sentence-transformers 6.0 连带升级 |
| `huggingface-hub` | 0.36.2 → 1.28.0 | 需 >=1.5.0 满足 transformers 5.x |

### 如何验证

```bash
cd HELLO
HF_ENDPOINT=https://hf-mirror.com python test_rag.py
```

- 测试 2 显示 `当前检索模式: dense`；语义查询「怎么让大模型少编造内容」跨字面命中 `rag_intro.md`（top1，分数 0.504）✅

### 可能踩的坑 / 排查

| 现象 | 原因 / 解决 |
|------|------------|
| `AttributeError: 'QdrantClient' object has no attribute 'search'` | qdrant-client >=1.10 移除了 `search()`，改用 `query_points()` |
| `ImportError: huggingface-hub>=1.5.0,<2.0 is required` | transformers 5.x 要求；`pip install "huggingface-hub>=1.5.0"` |
| 进程退出时 `QdrantClient.__del__` 报 `ImportError`（无害） | qdrant-client 本地嵌入式在解释器关闭时的清理时序问题，不影响功能 |

---

## [0.2.0] — 2026-08-25 — RAG 检索层升级（TF-IDF → Qdrant + Embedding）

### 一句话

检索层从「字符 n-gram 的 TF-IDF 稀疏检索」升级为「Qdrant + Embedding 稠密检索」，并保留 TF-IDF 作自动降级兜底。核心原则：**接口不变，只换实现**。

### 改动文件清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `app/core/rag.py` | 重写：新增 `TFIDFRetriever`、`QdrantRetriever`、`Embedder`/`SentenceTransformerEmbedder`/`OllamaEmbedder`、`build_embedder()`；保留 `RAGRetriever = TFIDFRetriever` 兼容别名 | 双检索实现 + embedding 抽象，`search()` 签名不变 |
| `app/core/agent_pipeline.py` | `get_retriever()` 改为 async 工厂 `_build_retriever()`（优先稠密、失败降级）；3 处调用改 `await`；新增 `current_retriever_mode()` | 自动降级 + 单例缓存 |
| `app/config.py` | 新增 `embedding_provider/model/base_url/api_key/device`、`qdrant_url/local_path`、`use_dense_retrieval` | 配置化 provider 与连接模式 |
| `requirements.txt` | 打开 `qdrant-client`、`sentence-transformers` | 稠密检索依赖 |
| `.env` / `.env.example` | 更新 embedding 与 Qdrant 配置 | 与 config 对齐 |
| `test_rag.py` | 新增「稠密语义检索 A/B 对照」测试 | 验证升级效果 + 降级 |
| `README.md` | 更新 RAG 说明、运行方式、验证结果、路线表 | 文档与代码一致 |

### 关键设计决策（面试可讲）

1. **延迟导入**：`sentence_transformers` / `qdrant_client` / `AsyncOpenAI` 都写在函数/方法内部，不在模块顶部 → 未安装时抛 ImportError，被工厂捕获 → 降级。
2. **工厂探活**：`_build_retriever` 里真正执行一次 `ingest_directory`（embedding + upsert），把「未装依赖 / 服务没起 / 模型失败」统一包进 try/except。
3. **三种 Qdrant 连接模式**：`url=`（Docker 服务端）、`path=`（本地嵌入式，无需 Docker）、`:memory:`（测试）。
4. **两种 embedding provider**：`sentence-transformers`（本地，推荐默认）、`ollama`（OpenAI 兼容 `/embeddings`）。

### 新增 / 变更的环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_PROVIDER` | `sentence-transformers` | embedding 后端（`sentence-transformers` / `ollama`） |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | ⚠️ 原为 `nomic-embed-text`，已改 |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | provider=ollama 时用 |
| `EMBEDDING_DEVICE` | `cpu` | sentence-transformers 推理设备 |
| `QDRANT_URL` | 空 | 空=本地嵌入式；填 `http://localhost:6333`=Docker 服务端 |
| `QDRANT_LOCAL_PATH` | `./qdrant_data` | 本地嵌入式数据目录 |
| `USE_DENSE_RETRIEVAL` | `true` | 总开关，false 则只用 TF-IDF |
| `QDRANT_HOST` / `QDRANT_PORT` | 保留 | 兼容旧配置，新代码已改用 `QDRANT_URL` |

### 如何验证

```bash
cd HELLO
python test_rag.py          # 4 项测试

# 快速看当前检索模式：
python -c "import asyncio; from app.core.agent_pipeline import get_retriever, current_retriever_mode; asyncio.run(get_retriever()); print(current_retriever_mode())"
```

- 输出 `dense` = 稠密检索已启用；`sparse` = 已降级（看启动日志里的 warning 判断原因）。

### 可能踩的坑 / 排查

| 现象 | 排查方向 |
|------|---------|
| 日志 warning「降级到 TF-IDF」 | 看 warning 里的异常类：`ModuleNotFoundError`→装依赖；连接错误→起 Qdrant/Ollama |
| 模型下载卡住 / 超时 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| 知识库改了但检索没变 | `get_retriever(force_reload=True)` 或删 `./qdrant_data` 目录 |
| 换 embedding 模型后维度对不上 | 无需改配置，`_ensure_collection` 按 `len(vectors[0])` 自动建集合 |
| 中文日志乱码 | Windows 控制台编码问题，脚本已 `reconfigure(encoding="utf-8")`，不影响功能 |

### 回滚

- 最快：`.env` 设 `USE_DENSE_RETRIEVAL=false`，即退回纯 TF-IDF。
- 彻底：恢复本次改动的 6 个文件（`rag.py`、`agent_pipeline.py`、`config.py`、`requirements.txt`、`.env`、`test_rag.py`）到上一版。

---

## [0.1.0] — 2026-08-16 — 初始版本（Day 8-12）

### 内容

- FastAPI 骨架 + 分层结构（`api` / `core` / `tools` / `models` / `config`）
- RAG 检索（离线 TF-IDF）
- SafetyTool（敏感词 + PII 正则）
- 模型A 生成 + 模型B 校验 + 双模型流水线
- `test_rag.py`、`test_dual_model.py` 端到端测试

### 验证

- `python test_rag.py`、`python test_dual_model.py` 实测通过（见 README 六）

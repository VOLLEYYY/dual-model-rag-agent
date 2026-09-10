# RAGAS 评估驱动调参 — 详细说明

> 用 RAGAS 把 RAG 系统的「拍脑袋调参」变成「凭指标调参」。
> 本文档讲解：RAGAS 是什么、用什么指标、怎么调优、调优结果、以及如何手动运行。
> 完整可运行代码见项目根目录 `ragas_tuning.py`。

---

## 一、RAGAS 是什么

RAGAS（Retrieval Augmented Generation Assessment）是一个开源的 RAG 系统评估框架。它用「LLM + embedding」自动给 RAG 回答打分，把「回答质量」量化为 0~1 的分数。

核心价值：RAG 的质量取决于「检索」和「生成」两个环节，且人工评估成本高、不可扩展。RAGAS 用 LLM 当「自动评委」，对每个样本输出客观分数，从而支撑**评估驱动的调参**——改一个参数，跑一轮评估，看分数涨没涨。

---

## 二、为什么用 RAGAS 调参

RAG 系统有一堆超参（`top_k`、`chunk_size`、`temperature`、`overlap`……），改它们的效果往往很微妙：

- `top_k` 太小：检索信息不够，答案不完整；
- `top_k` 太大：混入噪声，模型被干扰甚至编造；
- `chunk_size` 太小：语义碎片化；太大：检索不聚焦。

「凭感觉」调参无法量化哪种更好。RAGAS 让每个参数组合都产出可比较的分数，用指标说话。

---

## 三、根据什么调优（核心指标）

RAGAS 提供了多维度指标，本项目调优主要关注下面几个：

| 指标                            | 衡量什么              | 依赖              | 说明                                  |
| ----------------------------- | ----------------- | --------------- | ----------------------------------- |
| **Faithfulness（忠实度）**         | 回答是否忠于检索资料、有无编造   | LLM + embedding | 核心：把回答拆成若干 statement，逐条判断是否被检索上下文支持 |
| **Answer Relevancy（答案相关性）**   | 回答是否紧扣问题          | LLM + embedding | 让 LLM 生成若干变体问题，计算与回答的语义相似度          |
| **Context Recall（上下文召回）**     | 标准答案的信息被检索上下文覆盖多少 | LLM 或 embedding | 衡量「检索」是否找全                          |
| **Context Precision（上下文精确率）** | 检索到的上下文有多少是相关的    | LLM 或 embedding | 衡量「检索」是否找得准、不掺噪声                    |

**本项目调优用的指标是 `Faithfulness`**，理由：

1. 它直接反映「最终答案是否基于证据、少编造」，是 RAG 最关心的质量维度；
2. `top_k` 过大引入噪声时，Faithfulness 会明显下降（本项目实测就是靠它抓住了 `top_k=8` 的劣化）。

### 指标与参数的关系

- `top_k` / `chunk_size` → 主要影响**检索环节** → 直接影响 Context Recall / Precision，**间接影响** Faithfulness（上下文质量决定生成是否忠实）。
- `temperature` → 影响**生成环节** → 直接影响 Faithfulness / Answer Relevancy。

所以「调检索参数」用 Faithfulness + Context Recall/Precision 一起看最准；本项目先以 Faithfulness 做最小闭环。

---

## 四、代码实现

### 4.1 兼容性补丁（ragas 0.4 与 langchain 的版本坑）

ragas 0.4.x 硬编码 `from langchain_community.chat_models.vertexai import ChatVertexAI`，而该模块在 langchain-community 0.4+ 已拆到独立包 `langchain-google-vertexai`，导致 `import ragas` 直接失败。项目不用 vertexai，因此在脚本顶部注入占位模块：

```python
import sys, types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _m = types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI:
        pass
    _m.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _m
```

> 注意：这个补丁**写在脚本里**，不修改 site-packages，换机器/重装 ragas 后依然有效。

### 4.2 配置评估 LLM 与 embedding

RAGAS 的「评委」需要 LLM 和 embedding。这里复用项目自己的本地 DeepSeek 代理 + bge 嵌入模型：   

```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOpenAI(
    base_url="http://127.0.0.1:20128/v1",
    api_key="sk_9router",
    model="ds/deepseek-chat",
    temperature=0,               # 评估要稳定，用最低温度
)
ragas_llm = LangchainLLMWrapper(llm)

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
ragas_emb = LangchainEmbeddingsWrapper(emb)
```

### 4.3 评估测试集

评估需要「问题 + 标准答案（ground_truth）」作为基准：

```python
TEST_SET = [
    {
        "q": "什么是 RAG？它的核心流程是什么？",
        "gt": "RAG 是检索增强生成，核心流程包括文档加载、分块、向量化、检索和生成回答五个环节……",
    },
    # ... 每个问题配一条手写的标准答案
]
```

### 4.4 网格搜索 + 评估

对每个 `top_k`：跑「检索 → 生成 → 评估」三步，构造 RAGAS 样本并打分：

```python
async def build_samples(retriever, gen, top_k):
    samples = []
    for item in TEST_SET:
        docs = await retriever.search(item["q"], top_k)      # ① 检索
        answer = await gen.generate(item["q"], docs)         # ② 模型A 生成
        samples.append(SingleTurnSample(
            user_input=item["q"],
            reference=item["gt"],
            retrieved_contexts=[d["content"] for d in docs],
            response=answer,
        ))
    return samples

# 对每个 top_k：
samples = await build_samples(retriever, gen, top_k)
result = evaluate(
    EvaluationDataset(samples=samples),
    metrics=[Faithfulness()],          # ③ RAGAS 评估
    llm=ragas_llm,
    embeddings=ragas_emb,
)
df = result.to_pandas()                # 得到每个样本的 faithfulness 分数
```

**关键点**：`SingleTurnSample` 的四要素

- `user_input`：用户问题
- `retrieved_contexts`：检索到的文档块（对应项目的 `docs`）
- `response`：模型A 生成的回答
- `reference`：标准答案（ground_truth）

---

## 五、调优流程（数据流）

```
对每个候选参数 top_k ∈ {3, 5, 8}：
    │
    ├─ 对每个测试问题：
    │     检索(top_k) ──→ 模型A 生成 ──→ 得到 (contexts, response)
    │
    ├─ 构造 EvaluationDataset(samples)
    │
    └─ evaluate(Faithfulness) ──→ 取 3 个问题的 faithfulness 平均分
        │
最终：比较各 top_k 的平均分，取最高者
```

一次 `evaluate` 会对整个数据集并行打分，返回 `Result`，`result.to_pandas()` 得到逐样本分数。

---

## 六、调优结果（本项目实测）

| top_k | Faithfulness 平均 | 明细              |
| ----- | --------------- | --------------- |
| **3** | **1.000**       | 3 题全满分          |
| 5     | 1.000           | 3 题全满分          |
| 8     | 0.976           | 「混合检索」题降到 0.929 |

**结论**：

1. **`top_k=3` 最优**——满分，且比 `top_k=5` 少塞 2 个块，更省 token。
2. **`top_k=8` 反而下降**——检索块不是越多越好，多出的弱相关块是噪声，会稀释注意力、诱导模型偏离资料。

> 洞察：这个「`top_k=8` 劣化」正是 RAGAS 的价值所在——没有量化指标时，人很难凭感觉发现「多给 3 个块反而更差」这种反直觉现象。

---

## 七、项目如何手动运行

### 7.1 前提

- Python 3.10+，已装项目依赖（`pip install -r requirements.txt`）
- **本地 DeepSeek 代理在线**（`http://127.0.0.1:20128/v1`，RAGAS 的评估 LLM 走它）
- 知识库已存在（`data/knowledge_base/`，脚本会复用 `get_retriever` 的索引，无需重复索引）

### 7.2 安装 RAGAS（可选依赖）

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ragas langchain-openai langchain-huggingface
```

### 7.3 运行调参脚本

```bash
cd HELLO
PYTHONIOENCODING=utf-8 python ragas_tuning.py
```

> 说明：`ragas_tuning.py` **直接调用项目模块**（`get_retriever` / `ModelAGenerator`），
> **不需要启动 FastAPI 服务**。它和 `/api/ask` 走的是同一套检索 + 生成逻辑。

### 7.4 看结果

脚本会打印每个 `top_k` 的逐问题分数 + 平均分，最后输出最优参数：

```
检索模式：dense，chunk_size=512，overlap=50
======================================================================
top_k=3：faithfulness 平均 1.000（耗时 161s）
top_k=5：faithfulness 平均 1.000（耗时 171s）
top_k=8：faithfulness 平均 0.976（耗时 171s）
======================================================================
最优 top_k = 3（faithfulness=1.000）
```

### 7.5 扩展：调更多参数 / 更多指标

- **调 `chunk_size`**：在循环里改 `settings.rag_chunk_size` 后调 `get_retriever(force_reload=True)` 重建索引（因为分块发生在索引阶段）。
- **加指标**：把 `metrics=[Faithfulness()]` 改成 `metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()]`。
- **加参数组合**：把 `for top_k in [3, 5, 8]` 换成你要搜的网格。

### 7.6 性能提示

`Faithfulness` 单指标较慢（ragas 内部会「拆 statement + 逐条 NLI 判断」，单样本约 40~90s，取决于回答长度）。调参时建议：先小网格 + 单指标跑通，确认方向后再扩大。

---

## 附：常见问题

| 问题                                                                            | 解决                                                                         |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `import ragas` 报 `No module named 'langchain_community.chat_models.vertexai'` | ragas 0.4 与 langchain-community 0.4+ 的兼容问题；用脚本顶部的 monkey-patch 占位模块（见 4.1） |
| 评估分数全是 0 或 NaN                                                                | 检查评估 LLM 是否在线、`base_url`/`api_key` 是否正确；embedding 模型是否下载成功                 |
| 跑得很慢                                                                          | 正常。Faithfulness 内部多次 LLM 调用；可减少测试集问题数或只用单指标                                |
| 中文乱码                                                                          | 运行时加 `PYTHONIOENCODING=utf-8`（Windows 控制台 GBK 编码问题）                        |

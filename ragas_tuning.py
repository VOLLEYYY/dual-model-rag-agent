"""RAGAS 评估驱动的 RAG 参数调优（top_k）

用 RAGAS 的 Faithfulness（忠实度）指标，对检索参数 top_k 做网格搜索，
用「量化指标」取代「拍脑袋」，选出最优 top_k。

运行前提：
  - 本地 DeepSeek 代理在线（127.0.0.1:20128，评估 LLM）
  - 已 pip install ragas langchain-openai langchain-huggingface
  - 知识库已索引（脚本会复用 get_retriever 的单例缓存）

说明：
  - 本脚本只调 top_k（检索返回文档数）。chunk_size 会影响索引（需重建），
    若要一起调，需在循环里改 settings.rag_chunk_size 后 get_retriever(force_reload=True)。
  - Faithfulness 单指标较慢（ragas 内部多次 LLM 调用），完整调参可再加
    AnswerRelevancy / ContextRecall（见脚本底部注释）。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import types
import warnings

warnings.filterwarnings("ignore")

# ---- monkey-patch：修复 ragas 0.4.x 与 langchain-community 0.4+ 的兼容问题 ----
# ChatVertexAI 已从 langchain_community.chat_models.vertexai 拆到独立包，
# 本项目不用 vertexai，注入占位模块让 ragas 能正常 import。
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _m = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAI:
        pass

    _m.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _m

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.core.agent_pipeline import get_retriever, current_retriever_mode, ModelAGenerator


# ============================================================
#  评估测试集：问题 + 标准答案（ground_truth）
# ============================================================
TEST_SET = [
    {
        "q": "什么是 RAG？它的核心流程是什么？",
        "gt": (
            "RAG 是检索增强生成，核心流程包括文档加载、分块、向量化、检索和生成回答五个环节，"
            "让 LLM 先检索相关资料再基于资料生成回答，从而减少幻觉。"
        ),
    },
    {
        "q": "RAG 中为什么要做文档分块？分块大小怎么选？",
        "gt": (
            "分块把长文档切成小块便于精确检索，块太大语义不精确、太小语义不完整，"
            "块之间用 overlap 重叠防止关键信息被切断；通用推荐 chunk_size 500-1000、overlap 50-100。"
        ),
    },
    {
        "q": "什么是混合检索？它和单一向量检索有什么区别？",
        "gt": (
            "混合检索是结合关键字检索（稀疏）和向量检索（稠密）两种方式，按权重融合排序；"
            "单一向量检索对专有名词、缩写等精确匹配效果差，混合检索能兼顾语义匹配和精确匹配。"
        ),
    },
]


async def build_samples(retriever, gen, top_k: int):
    """对测试集每个问题：检索(top_k) + 生成回答，构造 RAGAS 样本。"""
    samples = []
    for item in TEST_SET:
        docs = await retriever.search(item["q"], top_k)
        answer = await gen.generate(item["q"], docs)
        samples.append(
            SingleTurnSample(
                user_input=item["q"],
                reference=item["gt"],
                retrieved_contexts=[d["content"] for d in docs],
                response=answer,
            )
        )
    return samples


async def main():
    # ---- 配置评估 LLM 与 embedding ----
    llm = ChatOpenAI(
        base_url=settings.model_a_base_url,
        api_key=settings.model_a_api_key or "sk_9router",
        model=settings.model_a_name,
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(llm)
    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    ragas_emb = LangchainEmbeddingsWrapper(emb)

    # ---- 复用项目检索器（单例缓存，不会重复索引） ----
    retriever = await get_retriever()
    gen = ModelAGenerator()
    print(f"检索模式：{current_retriever_mode()}，chunk_size={settings.rag_chunk_size}，overlap={settings.rag_chunk_overlap}")
    print("=" * 70)

    results = {}
    for top_k in [3, 5, 8]:
        t0 = time.time()
        samples = await build_samples(retriever, gen, top_k)
        # evaluate 是同步阻塞调用，放到线程池避免卡事件循环
        result = await asyncio.to_thread(
            evaluate,
            EvaluationDataset(samples=samples),
            [Faithfulness()],
            ragas_llm,
            ragas_emb,
        )
        df = result.to_pandas()
        results[top_k] = df
        mean_f = df["faithfulness"].mean()
        print(f"top_k={top_k}：faithfulness 平均 {mean_f:.3f}（耗时 {time.time()-t0:.0f}s）")
        for q, f in zip(TEST_SET, df["faithfulness"]):
            print(f"    - 「{q['q'][:20]}…」 {f:.3f}")

    # ---- 汇总对比 ----
    print("=" * 70)
    print("汇总（Faithfulness 越高，答案越忠于检索资料、越少编造）：")
    best_k, best_f = None, -1.0
    for top_k, df in results.items():
        f = df["faithfulness"].mean()
        print(f"  top_k={top_k}  →  {f:.3f}")
        if f > best_f:
            best_k, best_f = top_k, f
    print(f"\n最优 top_k = {best_k}（faithfulness={best_f:.3f}）")
    print("结论：可用此 top_k 作为生产默认值；若要更完整，加 AnswerRelevancy / ContextRecall 再跑一轮。")


if __name__ == "__main__":
    asyncio.run(main())

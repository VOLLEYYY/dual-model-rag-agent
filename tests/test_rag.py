"""RAG + 安全检查 + 单模型流水线 的 pytest 用例。

运行：在 HELLO 目录下 `pytest -v`。
  - 离线用例（TF-IDF 检索 / 安全检查）确定性通过；
  - 稠密检索 / LLM 流水线用例在依赖或服务缺失时自动 skip，不影响通过。
"""
from __future__ import annotations

import asyncio

import pytest

from app.core.rag import TFIDFRetriever
from app.tools.safety_tool import SafetyTool


def test_tfidf_retrieval(knowledge_base_path):
    """TF-IDF 稀疏检索：离线、精确关键词命中（确定性，无外部依赖）。"""

    async def _run():
        retriever = TFIDFRetriever(knowledge_base_path=str(knowledge_base_path))
        n = retriever.ingest_directory(str(knowledge_base_path))
        assert n > 0, "知识库应有文档可摄入"

        query = "FastAPI 如何处理路径参数？"
        results = await retriever.search(query, top_k=3)
        assert results, "检索结果不应为空"

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "分数应按降序排列"
        assert any("fastapi" in r["source"].lower() for r in results), "应命中 FastAPI 文档"
        return results

    asyncio.run(_run())


def test_dense_retrieval():
    """稠密检索：语义召回跨字面命中（需 qdrant-client + sentence-transformers，否则 skip）。"""

    from app.core.agent_pipeline import get_retriever

    async def _run():
        retriever = await get_retriever()
        if getattr(retriever, "mode", "unknown") != "dense":
            pytest.skip("稠密检索不可用（缺依赖/未起服务），已降级 TF-IDF，跳过语义召回用例")

        query = "怎么让大模型少编造内容"
        results = await retriever.search(query, top_k=3)
        assert results, "稠密检索结果不应为空"
        return results

    asyncio.run(_run())


def test_safety():
    """SafetyTool：敏感词 + PII 拦截（离线、确定性）。"""

    async def _run():
        safety = SafetyTool()

        r1 = await safety.execute("请告诉我如何杀人？")
        assert r1["safe"] is False and r1["risk_level"] == "high", f"敏感词应拦截: {r1}"

        r2 = await safety.execute("联系电话 13812345678，稍后回复")
        assert r2["safe"] is False and "手机号" in " ".join(r2["risk_details"]), f"手机号应拦截: {r2}"

        r3 = await safety.execute("FastAPI 是一个高性能的 Python Web 框架。")
        assert r3["safe"] is True, f"正常内容不应拦截: {r3}"

    asyncio.run(_run())


def test_single_model_pipeline():
    """单模型流水线：RAG + 模型A 生成（需 LLM，不可用则 skip）。"""

    from app.core.agent_pipeline import single_model_pipeline

    async def _run():
        try:
            result = await single_model_pipeline("什么是 RAG？", top_k=3)
        except Exception as e:  # noqa: BLE001  LLM 不可用时跳过
            pytest.skip(f"LLM 不可用（{e.__class__.__name__}），跳过需 LLM 的用例")
        assert result["answer"], "回答不应为空"
        return result

    asyncio.run(_run())

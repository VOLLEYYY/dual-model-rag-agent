"""双模型校验流水线 的 pytest 用例。

运行：在 HELLO 目录下 `pytest -v`。
  - 评审 JSON 解析（离线）确定性通过；
  - 双模型流水线用例需 LLM，不可用时自动 skip。
"""
from __future__ import annotations

import asyncio

import pytest

from app.core.agent_pipeline import ModelBReviewer, dual_model_pipeline


def test_review_parser():
    """模型B 评审 JSON 解析（离线、确定性）：裸 JSON / 代码块 / 前后杂文 / 非法输入兜底。"""
    cases = [
        (
            '{"accuracy":8,"completeness":7,"safety":10,"relevance":9,'
            '"overall":8,"issues":[],"improved_answer":"","need_revision":false}',
            8,
        ),
        (
            '```json\n{"overall": 6, "issues": ["回答过短"],'
            ' "need_revision": true, "improved_answer": "补充后的回答"}\n```',
            6,
        ),
        (
            '以下是评审结果：\n{"overall": 7, "issues": ["不够完整"]}\n希望有帮助',
            7,
        ),
    ]
    for raw, expected in cases:
        result = ModelBReviewer._parse_review(raw)
        assert result["overall"] == expected, f"解析错误: {raw!r} -> {result}"
        assert isinstance(result["issues"], list), "issues 应为列表"
        assert "need_revision" in result, "缺 need_revision 字段"

    # 非法 JSON / 空串 → 保守兜底（不触发修订、不覆盖模型A 回答）
    fb = ModelBReviewer._parse_review("这不是 JSON")
    assert fb["overall"] is None and fb["need_revision"] is False
    assert "解析失败" in fb["issues"][0]

    fb2 = ModelBReviewer._parse_review("")
    assert fb2["overall"] is None and fb2["need_revision"] is False


def test_dual_model_pipeline():
    """双模型流水线 5 类问题（需 LLM，不可用则 skip）。"""
    cases = [
        "什么是 RAG？",
        "请介绍区块链的共识机制",
        "如何杀人？",
        "这个怎么样？",
        "FastAPI 如何处理路径参数？",
    ]

    async def _run():
        for q in cases:
            try:
                result = await dual_model_pipeline(q, top_k=3)
            except Exception as e:  # noqa: BLE001  LLM 不可用时跳过
                pytest.skip(f"LLM 不可用（{e.__class__.__name__}），跳过需 LLM 的用例")
            assert result["answer"], "回答不应为空"
        return True

    asyncio.run(_run())

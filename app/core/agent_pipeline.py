"""Agent 核心流水线 — 双模型校验 RAG 问答

流水线架构（当前进度：Day 9-12）：

  用户提问
    → Step 1  RAG 检索   （Day 9  ✅；升级后 Qdrant 稠密检索，失败降级 TF-IDF）
    → Step 2  模型A 生成  （Day 11 ✅ OpenAI 兼容 LLM 生成）
    → Step 3  SafetyTool  （Day 10 ✅ 规则安全检查）
    → Step 4  模型B 校验  （Day 12 ✅ 独立模型质量评审）
    → 返回最终结果
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import settings
from ..models.schemas import AskResponse, SourceInfo
from ..tools.safety_tool import SafetyTool
from .llm import LLMClient
from .observability import observer
from .rag import QdrantRetriever, TFIDFRetriever, build_embedder


# ============================================================
#  检索器单例缓存：避免每次请求都重新分块/索引整个知识库
# ============================================================
logger = logging.getLogger("agent.pipeline")

_retriever: Optional[Any] = None
_retriever_mode: str = "sparse"   # "dense" | "sparse"


async def get_retriever(force_reload: bool = False):
    """获取（并惰性初始化）全局 RAG 检索器。

    优先使用 Qdrant 稠密检索；任何环节失败（未装依赖 / 未起服务 / 模型不可用）
    自动降级到 TF-IDF 稀疏检索。之后复用缓存。
    """
    global _retriever, _retriever_mode
    if _retriever is None or force_reload:
        _retriever, _retriever_mode = await _build_retriever(force_reload)
        logger.info("RAG 检索器就绪：mode=%s", _retriever_mode)
    return _retriever


async def _build_retriever(force_reload: bool = False):
    """构建检索器：优先 Qdrant 稠密检索，失败降级 TF-IDF。返回 (retriever, mode)。"""
    if settings.use_dense_retrieval:
        try:
            embedder = build_embedder(
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                device=settings.embedding_device,
            )
            retriever = QdrantRetriever(
                embedder=embedder,
                collection=settings.qdrant_collection,
                url=settings.qdrant_url or "",
                local_path=settings.qdrant_local_path or "",
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )
            n = await retriever.ingest_directory(
                settings.knowledge_base_path, force=force_reload
            )
            logger.info("稠密检索已就绪，摄入 %s 个 chunk", n)
            return retriever, "dense"
        except Exception as e:
            logger.warning(
                "Qdrant/embedding 不可用（%s: %s），降级到 TF-IDF 稀疏检索",
                e.__class__.__name__, e,
            )

    retriever = TFIDFRetriever(
        knowledge_base_path=settings.knowledge_base_path,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    retriever.ingest_directory(settings.knowledge_base_path)
    return retriever, "sparse"


def current_retriever_mode() -> str:
    """返回当前检索模式（"dense" 或 "sparse"），供测试/日志使用。"""
    return _retriever_mode


async def reindex_knowledge_base() -> Dict[str, Any]:
    """强制重建知识库索引：重新扫描目录 → 分块 → 向量化 → 写入 Qdrant。

    供「添加/修改知识库文档后」调用。因为检索器有单例缓存 + 幂等摄入，
    普通重启不会摄入新文档，必须 force 重建（见 CHANGELOG / RAGGG 指南）。
    返回检索模式、文档清单、分块数量，供前端展示。
    """
    from pathlib import Path

    kb = Path(settings.knowledge_base_path)
    documents: List[str] = []
    if kb.is_dir():
        documents = [
            f.name
            for f in sorted(kb.glob("*"))
            if f.suffix.lower() in (".md", ".txt")
        ]

    retriever = await get_retriever(force_reload=True)
    mode = current_retriever_mode()
    chunk_count = int(getattr(retriever, "chunk_count", 0))

    return {
        "success": True,
        "mode": mode,
        "documents": documents,
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "message": (
            f"索引重建完成：{len(documents)} 个文档 / {chunk_count} 个块"
            f"，检索模式 {mode}"
        ),
    }


# ============================================================
#  模型A：主回答生成模型
# ============================================================
class ModelAGenerator:
    """模型A：基于 OpenAI 兼容 LLM 的知识库问答生成器。"""

    _PROMPT = """你是一个知识库问答助手。请严格基于以下参考资料回答用户问题。

## 参考资料
{context_text}

## 用户问题
{question}

## 要求
1. 回答必须基于参考资料，不要编造信息。
2. 如果参考资料中没有相关信息，请明确回答"根据现有资料无法回答"。
3. 引用资料时请用【来源】标注出处。
"""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        resolved_url, resolved_key, resolved_model = settings.resolve_model_backend("a")
        self.llm = LLMClient(
            base_url=base_url or resolved_url,
            api_key=api_key if api_key is not None else resolved_key,
            model=model_name or resolved_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

    @staticmethod
    def _build_context(context: List[Dict[str, Any]]) -> str:
        if not context:
            return "（无参考资料）"
        return "\n\n---\n\n".join(
            f"【来源：{c.get('source', '未知')}】\n{c['content']}" for c in context
        )

    def _build_messages(self, question: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        prompt = self._PROMPT.format(
            question=question,
            context_text=self._build_context(context),
        )
        return [{"role": "user", "content": prompt}]

    async def generate(self, question: str, context: List[Dict[str, Any]]) -> str:
        """基于 RAG 检索到的上下文生成回答（一次性返回）。"""
        return await self.llm.invoke(self._build_messages(question, context))

    async def generate_stream(self, question: str, context: List[Dict[str, Any]]):
        """基于上下文流式生成回答（逐块产出，供 SSE 使用）。"""
        async for chunk in self.llm.stream(self._build_messages(question, context)):
            yield chunk


# ============================================================
#  模型B：质量校验模型
# ============================================================
def _review_fallback() -> Dict[str, Any]:
    """评审解析失败时的保守兜底：不触发修订，保留模型A 原始回答。"""
    return {
        "accuracy": None,
        "completeness": None,
        "safety": None,
        "relevance": None,
        "overall": None,
        "issues": ["评审结果解析失败"],
        "improved_answer": "",
        "need_revision": False,
    }


class ModelBReviewer:
    """模型B：独立质量评审模型（DeepSeek）。

    与模型A 使用不同模型，对模型A 的回答做「准确性 / 完整性 / 安全性 / 相关性」
    四维打分，输出结构化 JSON。这是「双模型制衡」的核心——不让模型自己评自己。
    """

    REVIEW_PROMPT = """你是一个严格的质量评审专家。请评审以下 Agent 回答的质量。

## 用户原始问题
{question}

## 参考资料（Agent 使用的上下文）
{context}

## Agent 回答
{answer}

## 评审维度（每项 0-10 分）
1. **准确性**：回答是否与参考资料一致？有无编造？
2. **完整性**：是否完整回答了用户的问题？
3. **安全性**：回答是否安全无害？
4. **相关性**：回答是否紧扣用户问题？

请只返回一个 JSON 对象，不要输出任何其他文字或 Markdown 代码块：
{{
    "accuracy": 8,
    "completeness": 7,
    "safety": 10,
    "relevance": 9,
    "overall": 8,
    "issues": ["问题1", "问题2"],
    "improved_answer": "改进后的回答（如无需改进则为空字符串）",
    "need_revision": false
}}
"""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        resolved_url, resolved_key, resolved_model = settings.resolve_model_backend("b")
        self.llm = LLMClient(
            base_url=base_url or resolved_url,
            api_key=api_key if api_key is not None else resolved_key,
            model=model_name or resolved_model,
            temperature=0.0,  # 评审要求稳定，用最低温度保证 JSON 稳定输出
            timeout=settings.llm_timeout,
        )

    @staticmethod
    def _build_context(context: List[Dict[str, Any]]) -> str:
        if not context:
            return "（无参考资料）"
        return "\n\n---\n\n".join(
            f"【来源：{c.get('source', '未知')}】\n{c['content']}" for c in context
        )

    @staticmethod
    def _parse_review(raw: str) -> Dict[str, Any]:
        """从模型输出中稳健解析评审 JSON（容忍代码块包裹、前后多余文字）。

        解析失败时返回保守兜底：不触发修订、不覆盖模型A 的回答。
        """
        if not raw:
            return _review_fallback()
        text = raw.strip()
        # 去掉 ```json ... ``` 或 ``` ... ``` 包裹
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # 截取第一个 { 到最后一个 } 之间的内容
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return _review_fallback()
        try:
            data = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return _review_fallback()
        if not isinstance(data, dict):
            return _review_fallback()
        # 归一化：缺失字段给默认值，避免下游 KeyError
        data.setdefault("issues", [])
        data.setdefault("need_revision", False)
        data.setdefault("improved_answer", "")
        data.setdefault("overall", None)
        return data

    async def review(
        self, question: str, answer: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """对模型A 的回答做质量评审，返回结构化评分 dict。"""
        prompt = self.REVIEW_PROMPT.format(
            question=question,
            answer=answer,
            context=self._build_context(context),
        )
        raw = await self.llm.invoke(
            [{"role": "user", "content": prompt}],
            max_tokens=settings.llm_review_max_tokens,
        )
        return self._parse_review(raw)


# ============================================================
#  单模型流水线（Day 11 验收）
# ============================================================
async def single_model_pipeline(question: str, top_k: int = 5) -> Dict[str, Any]:
    """RAG 检索 → 模型A 生成的单模型流水线。"""
    retriever = await get_retriever()
    generator = ModelAGenerator()

    docs = await retriever.search(question, top_k)
    answer = await generator.generate(question, docs)

    return {
        "question": question,
        "answer": answer,
        "sources": [d["source"] for d in docs],
    }


# ============================================================
#  双模型校验流水线（Day 12 验收）
# ============================================================
async def dual_model_pipeline(question: str, top_k: int = 5) -> Dict[str, Any]:
    """完整的双模型校验流水线：检索 → 生成 → 安检 → 校验 → 定稿。

    返回富结构 dict（含原始回答、评分、问题清单、安全结果、来源），
    供端到端测试与演示使用；HTTP 接口走 :func:`run_pipeline` 返回 AskResponse。
    """
    retriever = await get_retriever()
    safety = SafetyTool(sensitive_words=settings.get_safety_sensitive_words_list())
    model_a = ModelAGenerator()
    model_b = ModelBReviewer()

    trace: Dict[str, Any] = {}  # 预留：后续 Day 14 接入 Langfuse 追踪

    # Step 1: RAG 检索
    docs = await retriever.search(question, top_k)
    trace["rag_docs_count"] = len(docs)

    # Step 2: 模型A 生成
    answer = await model_a.generate(question, docs)
    trace["model_a_answer"] = answer

    # Step 3: 安全检查
    safety_result = await safety.execute(answer)
    trace["safety"] = safety_result
    if not safety_result["safe"]:
        return {
            "question": question,
            "answer": "回答因安全原因被拦截，未返回给用户。",
            "safety": safety_result,
            "sources": [],
        }

    # Step 4: 模型B 质量校验
    review = await model_b.review(question, answer, docs)
    trace["review"] = review

    # Step 5: 决定最终答案（需修订且给出改进版时，用改进版替换）
    if review.get("need_revision") and review.get("improved_answer"):
        final_answer = review["improved_answer"]
    else:
        final_answer = answer

    return {
        "question": question,
        "answer": final_answer,
        "original_answer": answer,  # 模型A 原始回答（用于对比）
        "review_score": review.get("overall"),
        "review_issues": review.get("issues", []),
        "need_revision": bool(review.get("need_revision", False)),
        "safety": safety_result,
        "sources": [d["source"] for d in docs],
        "trace": trace,
    }


# ============================================================
#  完整流水线入口
# ============================================================
def _docs_to_sources(docs: List[Dict[str, Any]]) -> List[SourceInfo]:
    """把检索结果转换为响应模型 SourceInfo 列表。"""
    return [
        SourceInfo(
            content=d["content"][:300],  # 截断，避免响应过大
            score=d["score"],
            source=d.get("source"),
        )
        for d in docs
    ]


def _to_score(value: Any) -> Optional[float]:
    """把评审总分稳健转为 float（截断到 0-10），非法时返回 None。"""
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, score))


def _format_review_comment(
    review: Dict[str, Any],
    overall: Optional[float],
    need_revision: bool,
) -> str:
    """把评审结果拼成人类可读的一句话，用于响应中的 review_comment。"""
    dims = []
    for key in ("accuracy", "completeness", "safety", "relevance"):
        v = review.get(key)
        if isinstance(v, (int, float)):
            dims.append(f"{key}={v}")
    parts: List[str] = []
    if dims:
        parts.append("维度评分 " + ", ".join(dims))
    if overall is not None:
        parts.append(f"总分 {overall}/10")
    if review.get("issues"):
        parts.append("问题: " + "; ".join(str(i) for i in review["issues"]))
    if need_revision:
        parts.append("已触发修订")
    return " | ".join(parts) if parts else "评审完成"


@observer.observe
async def run_pipeline(
    question: str,
    use_dual_model: bool = True,
    top_k: int = 5,
) -> AskResponse:
    """执行双模型校验 RAG 流水线。

    Args:
        question:       用户问题
        use_dual_model: 是否启用双模型校验（模型B 于 Day 12 实现）
        top_k:          检索文档数量

    Langfuse 可观测性：函数体被 @observer.observe 装饰，启用时自动创建一条 trace；
    各步骤关键数据通过 observer.update_trace 上报（未启用时为 no-op，零开销）。
    """
    observer.update_trace(
        name="ask",
        input={"question": question, "use_dual_model": use_dual_model, "top_k": top_k},
    )

    # ---- Step 1: RAG 检索 ----
    retriever = await get_retriever()
    docs = await retriever.search(question, top_k)
    observer.update_trace(
        metadata={
            "retriever_mode": current_retriever_mode(),
            "retrieved_docs": len(docs),
        }
    )

    # ---- Step 2: 模型A 生成 ----
    generator = ModelAGenerator()
    try:
        # 整体超时兜底：单次调用超时由 LLMClient 的 SDK timeout 控制，
        # 这里再包一层 wait_for，防止重试逻辑本身异常导致无限等待。
        answer = await asyncio.wait_for(
            generator.generate(question, docs),
            timeout=settings.llm_timeout,
        )
    except Exception as e:  # LLM 不可用/超时时的降级：返回检索到的相关资料片段
        snippet = docs[0]["content"] if docs else ""
        answer = (
            f"⚠️ 模型生成失败（{e.__class__.__name__}）。"
            f"以下为检索到的相关资料片段，仅供参考：\n\n{snippet}"
        )

    # ---- Step 3: SafetyTool 安全检查 ----
    safety = SafetyTool(sensitive_words=settings.get_safety_sensitive_words_list())
    safety_result = await safety.execute(answer)
    if settings.safety_enabled and not safety_result["safe"]:
        observer.update_trace(
            output={
                "answer": "（安全检查拦截）",
                "safety_passed": False,
                "risk_level": safety_result.get("risk_level"),
            },
        )
        return AskResponse(
            success=True,
            answer="回答因安全原因被拦截，未返回给用户。",
            review_score=None,
            review_comment=f"安全检查：{'; '.join(safety_result['risk_details'])}",
            safety_passed=False,
            sources=[],
        )

    # ---- Step 4: 模型B 质量校验（Day 12） ----
    review_score: Optional[float] = None
    review_comment: Optional[str] = None
    review_issues: List[str] = []
    need_revision: bool = False
    final_answer: str = answer
    if use_dual_model:
        try:
            reviewer = ModelBReviewer()
            review = await asyncio.wait_for(
                reviewer.review(question, answer, docs),
                timeout=settings.llm_timeout,
            )
            review_score = _to_score(review.get("overall"))
            review_issues = [str(i) for i in (review.get("issues") or [])]
            need_revision = bool(review.get("need_revision", False))
            if need_revision and review.get("improved_answer"):
                final_answer = str(review["improved_answer"])
            review_comment = _format_review_comment(review, review_score, need_revision)
        except Exception as e:  # 模型B 不可用时的降级：返回模型A 原始回答
            review_comment = (
                f"模型B 校验失败（{e.__class__.__name__}），返回模型A 原始回答。"
            )

    observer.update_trace(
        output={
            "answer": final_answer[:500],
            "safety_passed": True,
            "need_revision": need_revision,
            "review_score": review_score,
        },
        metadata={
            "sources": [d["source"] for d in docs],
        },
    )
    return AskResponse(
        success=True,
        answer=final_answer,
        original_answer=answer,
        review_score=review_score,
        review_comment=review_comment,
        review_issues=review_issues,
        need_revision=need_revision,
        safety_passed=True,
        sources=_docs_to_sources(docs),
    )


async def run_pipeline_stream(
    question: str,
    use_dual_model: bool = True,
    top_k: int = 5,
):
    """流式版流水线：逐步产出 SSE 事件 dict，供 /api/ask/stream 使用。

    事件序列：retrieving → retrieved → generating → token*(逐 token) →
              safety → review → done
    与 :func:`run_pipeline` 共享同一套降级策略：模型A / 模型B 失败时优雅降级。
    """
    # ---- Step 1: RAG 检索 ----
    yield {"type": "retrieving"}
    retriever = await get_retriever()
    docs = await retriever.search(question, top_k)
    yield {
        "type": "retrieved",
        "count": len(docs),
        "sources": [d["source"] for d in docs],
    }

    # ---- Step 2: 模型A 流式生成（逐 token 推送，降低首字延迟） ----
    generator = ModelAGenerator()
    yield {"type": "generating"}
    parts: List[str] = []
    try:
        async for token in generator.generate_stream(question, docs):
            parts.append(token)
            yield {"type": "token", "content": token}
    except Exception as e:  # 模型A 不可用时降级：返回检索到的资料片段
        fallback = (
            f"⚠️ 模型生成失败（{e.__class__.__name__}）。"
            f"以下为检索到的相关资料片段，仅供参考：\n\n"
            f"{docs[0]['content'] if docs else ''}"
        )
        parts = [fallback]
        yield {"type": "token", "content": fallback}
    answer = "".join(parts)

    # ---- Step 3: SafetyTool 安全检查 ----
    safety = SafetyTool(sensitive_words=settings.get_safety_sensitive_words_list())
    safety_result = await safety.execute(answer)
    yield {
        "type": "safety",
        "safe": safety_result["safe"],
        "risk_level": safety_result["risk_level"],
        "details": safety_result["risk_details"],
    }
    if settings.safety_enabled and not safety_result["safe"]:
        yield {
            "type": "done",
            "answer": "回答因安全原因被拦截，未返回给用户。",
            "safety_passed": False,
        }
        return

    # ---- Step 4: 模型B 质量校验 ----
    final_answer: str = answer
    review_score: Optional[float] = None
    review_comment: Optional[str] = None
    review_issues: List[str] = []
    need_revision: bool = False
    if use_dual_model:
        try:
            reviewer = ModelBReviewer()
            review = await reviewer.review(question, answer, docs)
            review_score = _to_score(review.get("overall"))
            review_issues = [str(i) for i in (review.get("issues") or [])]
            need_revision = bool(review.get("need_revision", False))
            if need_revision and review.get("improved_answer"):
                final_answer = str(review["improved_answer"])
            review_comment = _format_review_comment(review, review_score, need_revision)
        except Exception as e:  # 模型B 不可用时降级：返回模型A 原始回答
            review_comment = (
                f"模型B 校验失败（{e.__class__.__name__}），返回模型A 原始回答。"
            )
    yield {
        "type": "review",
        "score": review_score,
        "need_revision": need_revision,
        "issues": review_issues,
        "comment": review_comment,
    }

    yield {
        "type": "done",
        "answer": final_answer,
        "original_answer": answer,
        "safety_passed": True,
        "sources": [d["source"] for d in docs],
    }

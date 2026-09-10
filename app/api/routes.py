"""API 路由定义 — 对外暴露的 HTTP 端点"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..config import get_settings
from ..models.schemas import AskRequest, AskResponse, HealthResponse, ErrorResponse, ReindexResponse
from ..core.agent_pipeline import run_pipeline, run_pipeline_stream, reindex_knowledge_base

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点 — Docker / K8s 探针使用"""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """双模型校验问答端点

    流水线: RAG 检索 → 模型A 生成 → 安全检查 → 模型B 校验 → 返回
    """
    try:
        response = await run_pipeline(
            question=request.question,
            use_dual_model=request.use_dual_model,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"流水线执行失败: {str(e)}"
        )


@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """SSE 流式问答端点：逐步推送 检索→生成→安检→校验 事件。

    与 /ask 共享同一套流水线（run_pipeline_stream），区别是逐 token 推送、
    降低首字延迟（TTFT），适用于聊天式交互。
    """
    async def event_gen():
        try:
            async for event in run_pipeline_stream(
                question=request.question,
                use_dual_model=request.use_dual_model,
                top_k=request.top_k,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/knowledge/reindex", response_model=ReindexResponse)
async def reindex_knowledge():
    """重建知识库索引（添加/修改知识库文档后调用）。

    重新扫描 data/knowledge_base/，分块、向量化、写入 Qdrant。
    因检索器有单例缓存 + 幂等摄入，普通重启不会摄入新文档，需此端点 force 重建。
    """
    try:
        result = await reindex_knowledge_base()
        return ReindexResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建索引失败: {str(e)}")

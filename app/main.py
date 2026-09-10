"""FastAPI 应用入口

启动方式:
    # 开发模式
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

    # 或使用根目录 run.py
    python run.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings, settings
from .api.routes import router


def create_app() -> FastAPI:
    """FastAPI 应用工厂"""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="双模型校验本地知识库 Agent — RAG 检索 + 模型A 生成 + 模型B 独立评审（OpenAI 兼容后端，可切换 DeepSeek / Ollama）",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ---- CORS 中间件 ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- 注册路由 ----
    app.include_router(router, prefix="/api")

    # ---- 启动事件 ----
    @app.on_event("startup")
    async def startup():
        line = "=" * 60
        print(f"\n{line}")
        print(f"  {settings.app_name} v{settings.app_version}")
        print(f"{line}")
        print(f"  API Docs: http://localhost:{settings.port}/docs")
        print(f"  ReDoc:    http://localhost:{settings.port}/redoc")
        print(f"  Health:   http://localhost:{settings.port}/api/health")
        print(f"{line}\n")

    @app.on_event("shutdown")
    async def shutdown():
        print("\n  App shutting down...\n")

    # ---- 根路径 ----
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": "/docs",
        }

    @app.get("/health")
    async def health():
        """根级别健康检查（兼容简单探针）"""
        return {"status": "ok"}

    return app


# ---- 应用实例 ----
app = create_app()

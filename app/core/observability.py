"""Langfuse 可观测性封装（Day 14，可选依赖）

设计原则（与项目「优雅降级」一致）：
- langfuse 是可选依赖，未安装或未配置密钥时自动禁用，主流程不受影响。
- 通过 .env 配置（LANGFUSE_ENABLED / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL）。
- 只在编排层（agent_pipeline）打点，不侵入 rag / llm 等底层模块，保持分层干净。

基于 langfuse SDK v3（注意：v4 移除了 langfuse.decorators 与 update_current_trace，
本项目按 v3 的「observe 装饰器 + get_client().update_current_trace」编写）：

    from langfuse import observe, get_client

    @observe()                       # 装饰器自动创建 trace + span
    async def run_pipeline(...):
        client.update_current_trace(name=..., input=..., output=..., metadata=...)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

from ..config import settings

logger = logging.getLogger("agent.observability")


class LangfuseObserver:
    """Langfuse 追踪门面：未启用时所有方法都是 no-op（零开销降级）。"""

    def __init__(self) -> None:
        self._enabled: bool | None = None
        self._client: Any = None

    # ------------------------------------------------------------
    # 初始化 / 开关
    # ------------------------------------------------------------
    def is_enabled(self) -> bool:
        """惰性判断是否启用（只初始化一次）。"""
        if self._enabled is None:
            self._enabled = self._try_enable()
        return self._enabled

    def _try_enable(self) -> bool:
        """尝试启用：需要「开关开启 + 密钥齐全 + langfuse 已安装」三者同时满足。"""
        if not settings.langfuse_enabled:
            return False
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.info("Langfuse 未配置密钥，可观测性已禁用（不影响主流程）")
            return False
        try:
            import langfuse  # noqa: F401  延迟导入：可选依赖
        except ImportError:
            logger.info("langfuse 未安装，可观测性已禁用")
            return False
        # 在 SDK 初始化前注入环境变量（langfuse 通过 LANGFUSE_* 环境变量读配置）
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)
        try:
            from langfuse import get_client
            self._client = get_client()
        except Exception as e:
            logger.warning("Langfuse 客户端初始化失败（%s），可观测性已禁用", e)
            return False
        logger.info("Langfuse 可观测性已启用：%s", settings.langfuse_base_url)
        return True

    # ------------------------------------------------------------
    # 装饰器
    # ------------------------------------------------------------
    def observe(self, func: Callable) -> Callable:
        """装饰器：启用时用 langfuse 的 @observe() 包装，否则原样返回（no-op）。"""
        if not self.is_enabled():
            return func
        try:
            from langfuse import observe as _observe
            return _observe()(func)
        except Exception as e:  # 追踪失败绝不能影响主流程
            logger.warning("Langfuse @observe 装饰失败（%s），跳过追踪", e)
            return func

    # ------------------------------------------------------------
    # 上下文记录（仅在 @observe 包装的函数体内有效）
    # ------------------------------------------------------------
    def update_trace(self, **kwargs: Any) -> None:
        """更新当前 trace 的元数据（name / input / output / metadata 等）。"""
        if not self.is_enabled():
            return
        try:
            self._client.update_current_trace(**kwargs)
        except Exception:  # 任何异常都静默吞掉，不影响业务
            pass

    def update_span(self, **kwargs: Any) -> None:
        """更新当前 span 的元数据（供分步打点使用）。"""
        if not self.is_enabled():
            return
        try:
            self._client.update_current_span(**kwargs)
        except Exception:
            pass

    def flush(self) -> None:
        """主动刷出缓存 trace（进程退出前调用，避免上报延迟）。"""
        if not self.is_enabled():
            return
        try:
            self._client.flush()
        except Exception:
            pass


# 全局单例
observer = LangfuseObserver()

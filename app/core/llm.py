"""LLM 调用封装 — 统一 OpenAI 兼容接口

本模块把「模型调用」从业务逻辑中解耦，业务代码只依赖 :class:`LLMClient`，
因此可以无缝切换后端（均走 OpenAI 兼容协议）：
  - 本地 DeepSeek API（默认，http://127.0.0.1:20128/v1）
  - Ollama（http://localhost:11434/v1，同样提供 OpenAI 兼容接口）
  - 其他 OpenAI 兼容服务（vLLM、硅基流动、DashScope 等）
"""
from __future__ import annotations

from typing import AsyncIterator, Dict, List

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


# ============================================================
#  模型调用健壮性：重试 + 指数退避（Day 18）
# ============================================================
_RETRY_ATTEMPTS = 3      # 最多重试次数（不含首次调用）
_RETRY_MIN_WAIT = 1.0    # 指数退避起始等待（秒）
_RETRY_MAX_WAIT = 10.0   # 指数退避上限（秒）


def _is_retryable_error(exc: BaseException) -> bool:
    """判断异常是否值得重试。

    可重试：连接失败 / 超时 / 限流（429）/ 5xx 服务端错误 —— 这些通常是瞬时的，
    指数退避后重试大概率成功。
    不可重试：4xx 客户端错误（参数错误、鉴权失败等）—— 重试也白费，直接放弃。
    """
    if isinstance(
        exc,
        (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError),
    ):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 500 <= status < 600


class LLMClient:
    """OpenAI 兼容 LLM 客户端封装，提供一次性生成与流式生成两种能力。"""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "ds/deepseek-chat",
        timeout: float = 120.0,
        temperature: float = 0.1,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        # 本地服务通常不校验 key，但 OpenAI SDK 要求非空，故给占位值
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            # 关闭 SDK 内部重试：重试统一由 tenacity 控制（invoke 上），
            # 避免「SDK 重试 × tenacity 重试」叠加导致耗时不可控。
            max_retries=0,
        )

    @retry(
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,  # 重试耗尽后抛出原始异常，交由上层降级逻辑处理
    )
    async def invoke(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        """一次性生成，返回完整回答文本。

        带重试：连接失败 / 超时 / 限流 / 5xx 等瞬时错误会自动重试（指数退避），
        最多 {_RETRY_ATTEMPTS} 次；重试耗尽后抛原始异常，由流水线捕获降级。
        """
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    async def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """流式生成，逐块产出回答文本（供 SSE 接口使用）。"""
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

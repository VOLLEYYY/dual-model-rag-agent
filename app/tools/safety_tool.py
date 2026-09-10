"""安全检查工具 — 检测回答中的有害/敏感内容

在模型A 生成回答后、返回用户前执行（Day 10 实现）。

采用规则引擎实现，包含两类检测：
  1. 敏感词匹配 —— 内置默认词表 + 可在配置 `SAFETY_SENSITIVE_WORDS` 中追加
  2. PII 正则检测 —— 手机号、身份证号、银行卡号、邮箱

说明：这是「规则引擎示例」，确定性、离线、可测试。
生产环境可替换为基于 LLM 的安全审核（如路线图 Day 10 提到的 R-Judge 思路）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# 内置默认敏感词（示例词表，可按需扩充）
DEFAULT_SENSITIVE_WORDS: List[str] = [
    "炸弹制作",
    "制造炸药",
    "如何杀人",
    "购买枪支",
    "毒品配方",
    "自杀方法",
]

# PII 检测规则：标签 -> 正则
_PII_PATTERNS: List[tuple] = [
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("银行卡号", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("邮箱地址", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


class SafetyTool:
    """安全检查工具。

    在流水线中位于「模型A 生成」之后：若检测到不安全内容，则拦截回答，
    不向用户返回可能有害或泄露隐私的文本。
    """

    name = "safety_check"
    description = "检测文本中的有害/敏感内容，返回是否安全及风险详情"

    def __init__(self, sensitive_words: List[str] | None = None):
        # 传入空列表时使用内置默认词表；传入自定义词表则与之合并
        self.sensitive_words = list(
            set(DEFAULT_SENSITIVE_WORDS) | set(sensitive_words or [])
        )

    def _check_sensitive_words(self, text: str) -> List[str]:
        return [w for w in self.sensitive_words if w in text]

    def _check_pii(self, text: str) -> List[str]:
        return [label for label, pattern in _PII_PATTERNS if pattern.search(text)]

    async def execute(self, text: str) -> Dict[str, Any]:
        """执行安全检查。

        Returns:
            dict: {
                "safe": bool,            # 是否安全
                "risk_level": str,       # "low" | "medium" | "high"
                "risk_details": list,    # 命中的风险描述
            }
        """
        sensitive_hits = self._check_sensitive_words(text)
        pii_hits = self._check_pii(text)

        details: List[str] = []
        if sensitive_hits:
            details.append(f"命中敏感词: {', '.join(sensitive_hits)}")
        if pii_hits:
            details.append(f"疑似泄露个人信息: {', '.join(pii_hits)}")

        if sensitive_hits:
            risk_level = "high"
        elif pii_hits:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "safe": not details,
            "risk_level": risk_level,
            "risk_details": details,
        }

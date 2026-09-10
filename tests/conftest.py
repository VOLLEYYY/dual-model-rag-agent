"""pytest 公共配置：sys.path + 输出编码 + 共享 fixture。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 项目根目录（HELLO）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 保证 `from app.core.xxx import ...` 可导入（app 包在 HELLO/app/）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台统一 UTF-8 输出，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录（HELLO）绝对路径。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def knowledge_base_path() -> Path:
    """知识库目录绝对路径。"""
    return PROJECT_ROOT / "data" / "knowledge_base"

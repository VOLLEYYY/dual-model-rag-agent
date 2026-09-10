"""请求/响应数据模型 — 基于 Pydantic 的参数校验与序列化"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
#  请求模型
# ============================================================

class AskRequest(BaseModel):
    """问答请求"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="用户问题",
        example="什么是 RAG？"
    )
    use_dual_model: bool = Field(
        default=True,
        description="是否启用双模型校验（模型A生成 + 模型B校验）"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="检索文档数量"
    )


# ============================================================
#  响应模型
# ============================================================

class SourceInfo(BaseModel):
    """检索来源信息"""
    content: str = Field(..., description="文档片段内容")
    score: float = Field(..., description="相似度得分")
    source: Optional[str] = Field(default=None, description="来源文件名")


class AskResponse(BaseModel):
    """问答响应"""
    success: bool = Field(..., description="是否成功")
    answer: str = Field(default="", description="最终回答")
    original_answer: Optional[str] = Field(
        default=None,
        description="模型A 原始回答（模型B 修订前的回答）"
    )
    review_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="模型B 质量评分"
    )
    review_comment: Optional[str] = Field(
        default=None,
        description="模型B 评审意见"
    )
    review_issues: List[str] = Field(
        default_factory=list,
        description="模型B 发现的回答问题清单"
    )
    need_revision: bool = Field(
        default=False,
        description="模型B 是否建议修订"
    )
    safety_passed: Optional[bool] = Field(
        default=None,
        description="安全检查是否通过"
    )
    sources: List[SourceInfo] = Field(
        default_factory=list,
        description="检索到的文档来源"
    )


class ReindexResponse(BaseModel):
    """知识库索引重建响应"""
    success: bool = Field(..., description="是否成功")
    mode: str = Field(..., description="检索模式 dense/sparse")
    documents: List[str] = Field(default_factory=list, description="知识库文档文件名列表")
    document_count: int = Field(default=0, description="文档数量")
    chunk_count: int = Field(default=0, description="分块数量")
    message: str = Field(default="", description="结果说明")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")


class ErrorResponse(BaseModel):
    """通用错误响应"""
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")

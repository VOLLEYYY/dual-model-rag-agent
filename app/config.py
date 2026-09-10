"""配置管理模块 — 基于 Pydantic Settings 的统一配置

所有配置项通过环境变量或 .env 文件设置，优先级：环境变量 > .env > 默认值
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env 文件（项目根目录）
load_dotenv()


class Settings(BaseSettings):
    """应用全局配置"""

    # ========== 应用基本信息 ==========
    app_name: str = "双模型校验知识库 Agent"
    app_version: str = "0.5.4"
    debug: bool = False

    # ========== 服务器配置 ==========
    host: str = "0.0.0.0"
    port: int = 8000

    # ========== CORS 配置 ==========
    cors_origins: str = "http://localhost:7860,http://127.0.0.1:7860,http://localhost:3000"

    # ========== LLM 配置（模型A：主生成模型） ==========
    # 默认指向本地 OpenAI 兼容 DeepSeek API；改用 Ollama 时填 http://localhost:11434/v1 + qwen2.5:7b
    model_a_api_key: str = ""
    model_a_base_url: str = "http://127.0.0.1:20128/v1"
    model_a_name: str = "ds/deepseek-chat"

    # ========== LLM 配置（模型B：质量校验模型） ==========
    model_b_api_key: str = ""
    model_b_base_url: str = "http://127.0.0.1:20128/v1"
    model_b_name: str = "ds/deepseek-chat"

    # ========== 模型A/B 后端 provider（可切换） ==========
    # 两种 provider（model_provider 切换），与 embedding 的 EMBEDDING_PROVIDER 对称：
    #   deepseek —— 本地 OpenAI 兼容 DeepSeek 代理（默认，用 model_a_* / model_b_*）
    #   ollama   —— 本地 Ollama 服务（用 ollama_*，需先 ollama pull 对应模型）
    model_provider: str = "deepseek"
    ollama_base_url: str = "http://localhost:11434/v1"   # provider=ollama 时的服务地址
    ollama_model_a: str = "qwen2.5:7b"                   # provider=ollama 时模型A 的模型名
    ollama_model_b: str = "qwen2.5:7b"                   # provider=ollama 时模型B 的模型名

    # ========== LLM 通用配置 ==========
    llm_timeout: float = 120.0      # 单次调用超时（秒）
    llm_temperature: float = 0.1    # 生成温度（越低越稳定）
    # 模型B 评审输出 token 上限：给足余量，避免评审 JSON 被截断为空。
    llm_review_max_tokens: int = 4096

    # ========== 向量嵌入模型 ==========
    # 两种 provider（embedding_provider 切换）：
    #   sentence-transformers —— 本地推理，无需外部服务（推荐默认，适合无 Docker/Ollama 环境）
    #   ollama —— 走 OpenAI 兼容 /embeddings 接口，需本机跑 Ollama
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"        # 中文语义检索，小模型本地快速跑
    embedding_base_url: str = "http://localhost:11434/v1"  # provider=ollama 时的服务地址
    embedding_api_key: str = ""                            # provider=ollama 时通常留空
    embedding_device: str = "cpu"                          # sentence-transformers 推理设备

    # ========== HuggingFace 模型下载镜像 ==========
    # 国内访问 huggingface.co 慢/不通，默认走 hf-mirror 镜像；
    # 见文件底部 os.environ.setdefault，未显式设置时自动注入。
    hf_endpoint: str = "https://hf-mirror.com"

    # ========== Qdrant 向量数据库 ==========
    # 两种连接模式（qdrant_url 优先，否则本地嵌入式）：
    #   1) 本地嵌入式（默认，无需 Docker）：qdrant_url 留空，数据落到 qdrant_local_path
    #   2) 服务端（Docker/生产）：设置 qdrant_url，如 http://localhost:6333
    qdrant_url: str = ""                                   # 服务端地址；空则用本地嵌入式
    qdrant_local_path: str = "./qdrant_data"               # 本地嵌入式存储目录（无需 Docker）
    qdrant_collection: str = "knowledge_base"
    # 兼容保留（服务端模式，Docker 编排用）：
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ========== 检索模式 ==========
    # True = 优先 Qdrant 稠密检索，任何环节失败自动降级 TF-IDF 稀疏检索
    use_dense_retrieval: bool = True

    # ========== RAG 配置 ==========
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5

    # ========== 知识库路径 ==========
    knowledge_base_path: str = "./data/knowledge_base"

    # ========== 安全检查配置 ==========
    safety_enabled: bool = True
    safety_sensitive_words: str = ""   # 逗号分隔的敏感词列表

    # ========== Langfuse 可观测性（Day 14，可选） ==========
    # langfuse 是可选依赖：未安装或未配置密钥时，observability 层自动禁用，主流程不受影响。
    # 密钥在 Langfuse 面板 Project Settings → API Keys 生成（pk-lf-... / sk-lf-...）；
    # LANGFUSE_BASE_URL 按注册时选择的数据区域填（日本地区 = https://jp.cloud.langfuse.com）。
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""   # pk-lf-...
    langfuse_secret_key: str = ""   # sk-lf-...
    langfuse_base_url: str = "https://jp.cloud.langfuse.com"

    # ========== 日志配置 ==========
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
        # 字段以 model_ 开头（如 model_a_name）与 Pydantic 保留命名空间冲突，此处关闭该限制
        protected_namespaces = ()

    def get_cors_origins_list(self) -> List[str]:
        """解析 CORS origins 为列表"""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def get_safety_sensitive_words_list(self) -> List[str]:
        """解析敏感词列表"""
        if not self.safety_sensitive_words:
            return []
        return [w.strip() for w in self.safety_sensitive_words.split(",") if w.strip()]

    def resolve_model_backend(self, role: str):
        """按 model_provider 解析模型A/B 的后端，返回 (base_url, api_key, model_name)。

        role: "a"（生成模型）或 "b"（校验模型）。
        provider=ollama 时走 Ollama（忽略 model_a_* / model_b_* 的 base_url/name）；
        否则走 DeepSeek 代理（默认）。
        """
        if self.model_provider == "ollama":
            model = self.ollama_model_a if role == "a" else self.ollama_model_b
            return self.ollama_base_url, "", model
        if role == "a":
            return self.model_a_base_url, self.model_a_api_key, self.model_a_name
        return self.model_b_base_url, self.model_b_api_key, self.model_b_name


# ---- 全局单例 ----
settings = Settings()

# 让 huggingface_hub / sentence-transformers 走镜像（国内加速）。
# 必须在延迟导入 sentence_transformers 之前设置，否则首次加载 bge 模型
# 会连官方 huggingface.co 超时重试、阻塞事件循环。
# 用 setdefault：尊重用户已在系统环境显式设置的 HF_ENDPOINT。
os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)


def get_settings() -> Settings:
    """获取配置实例（依赖注入用）"""
    return settings

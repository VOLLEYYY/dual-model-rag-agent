"""RAG 检索器 — 文档分块 + 向量化 + 相似度检索

本模块提供两套检索实现，共享同一个 `search(query, top_k)` 异步接口：

1. `TFIDFRetriever` —— 稀疏检索（兜底 / 精确关键词通道）
   自包含、零外部依赖：字符 n-gram 的 TF-IDF + 余弦相似度，线性扫描。

2. `QdrantRetriever` —— 稠密检索（生产主力）
   embedding 模型把文本映射为稠密向量，存入 Qdrant 向量库，
   用 HNSW 近似最近邻检索，支持 payload 过滤，百万级 chunk 无压力。

二者「接口不变，只换实现」：调用方只依赖
    `await search(query, top_k) -> [{content, score, source}]`
因此上层流水线无需改动即可在两种检索间切换，或做降级。
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("agent.rag")


# ============================================================
#  共享工具：分词 + 分块
# ============================================================
def _tokenize(text: str) -> List[str]:
    """把文本切成 token：英文/数字按词，中文按单字 + 相邻 bigram。

    中文同时保留单字与 bigram，是为了兼顾「精确命中」与「短语语义」，
    例如「向量检索」既能命中「向量」也能命中「向量检索」这一整体。
    """
    text = text.lower()
    tokens: List[str] = []
    tokens.extend(m.group() for m in re.finditer(r"[a-z0-9]+", text))
    cjk = re.findall(r"[一-鿿]", text)
    tokens.extend(cjk)
    tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def _split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """定长字符分块，overlap 防止关键信息被切在边界。"""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


# ============================================================
#  Embedder：把文本映射为稠密向量
# ============================================================
class Embedder(ABC):
    """embedding 抽象接口。上层只依赖 embed_texts/embed_query，不关心具体后端。"""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量把文本映射为稠密向量，返回与输入等长的向量列表。"""

    async def embed_query(self, text: str) -> List[float]:
        """把单条查询文本映射为向量。"""
        vecs = await self.embed_texts([text])
        return vecs[0]


class SentenceTransformerEmbedder(Embedder):
    """本地 sentence-transformers 嵌入（无需外部服务，需 `pip install sentence-transformers`）。

    模型在 __init__ 时加载；若未安装依赖或模型名错误，会在构造时抛 ImportError/异常，
    由上层工厂捕获后降级到 TF-IDF。
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # 延迟导入：未安装则抛 ImportError
        self._model = SentenceTransformer(model_name, device=device)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # encode 是 CPU/GPU 密集型同步调用，放到线程池避免阻塞事件循环
        vectors = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]


class OllamaEmbedder(Embedder):
    """Ollama 嵌入（走 OpenAI 兼容 /embeddings 接口，需本机跑 Ollama）。"""

    def __init__(self, model: str, base_url: str, api_key: str = ""):
        from openai import AsyncOpenAI  # 延迟导入
        self._model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


def build_embedder(
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    device: str = "cpu",
) -> Embedder:
    """按 provider 构建 embedder。未知 provider 抛 ValueError。"""
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model, device=device)
    if provider == "ollama":
        return OllamaEmbedder(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(
        f"未知 embedding provider: {provider}（可选 sentence-transformers / ollama）"
    )


# ============================================================
#  TFIDFRetriever：稀疏检索（兜底）
# ============================================================
class TFIDFRetriever:
    """基于 TF-IDF 的本地稀疏检索器。

    职责：把知识库文档分块 → 统计词频建立 TF-IDF 索引，并对用户问题做余弦相似度检索。
    """

    mode = "sparse"

    def __init__(
        self,
        knowledge_base_path: str = "./data/knowledge_base",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.kb_path = Path(knowledge_base_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: List[Dict[str, Any]] = []   # 每项 {"content", "source", "tf"}
        self._doc_freq: Counter = Counter()        # token -> 出现该 token 的 chunk 数

    def _tfidf(self, tf: Counter) -> Dict[str, float]:
        """由词频计算 TF-IDF 稀疏向量。"""
        total = sum(tf.values()) or 1
        vec: Dict[str, float] = {}
        for token, cnt in tf.items():
            # 平滑 IDF：某 token 越罕见，权重越高
            idf = math.log((len(self._chunks) + 1) / (self._doc_freq[token] + 1)) + 1
            vec[token] = (cnt / total) * idf
        return vec

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = a.keys() & b.keys()
        dot = sum(a[k] * b[k] for k in common)
        if dot == 0:
            return 0.0
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb)

    def ingest_document(self, file_path: str) -> int:
        """摄入单个文档（.md/.txt），分块并建立索引，返回新增块数。"""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        added = 0
        for chunk in _split(text, self.chunk_size, self.chunk_overlap):
            tf = Counter(_tokenize(chunk))
            if not tf:
                continue
            self._chunks.append({"content": chunk, "source": path.name, "tf": tf})
            self._doc_freq.update(tf.keys())
            added += 1
        return added

    def ingest_directory(self, dir_path: str) -> int:
        """摄入目录下所有 .md/.txt 文档，返回总块数。目录不存在时返回 0。"""
        path = Path(dir_path)
        if not path.is_dir():
            return 0
        total = 0
        for f in sorted(path.glob("*")):
            if f.suffix.lower() in (".md", ".txt"):
                total += self.ingest_document(str(f))
        return total

    @property
    def chunk_count(self) -> int:
        """已索引的文档块数量。"""
        return len(self._chunks)

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索与 query 最相似的 top_k 个文档块，按相似度降序返回。"""
        if not self._chunks:
            return []
        qvec = self._tfidf(Counter(_tokenize(query)))
        scored: List[tuple] = []
        for chunk in self._chunks:
            sim = self._cosine(qvec, self._tfidf(chunk["tf"]))
            if sim > 0:
                scored.append((sim, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content": chunk["content"],
                "score": round(sim, 4),
                "source": chunk["source"],
            }
            for sim, chunk in scored[:top_k]
        ]


# 向后兼容别名：旧代码 `from .rag import RAGRetriever` 依然可用
RAGRetriever = TFIDFRetriever


# ============================================================
#  QdrantRetriever：稠密检索
# ============================================================
class QdrantRetriever:
    """基于 Qdrant + embedding 的稠密检索器。

    连接模式（url 优先，其次 local_path，最后内存）：
      - url="http://localhost:6333"  服务端模式（Docker/生产）
      - local_path="./qdrant_data"   本地嵌入式模式（无需 Docker，数据落盘）
      - 二者皆空                    内存模式（测试用，重启即失）
    """

    mode = "dense"

    def __init__(
        self,
        embedder: Embedder,
        collection: str = "knowledge_base",
        url: str = "",
        local_path: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.embedder = embedder
        self.collection = collection
        self.url = url
        self.local_path = local_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._client = None

    @property
    def _qdrant(self):
        """惰性创建 Qdrant 客户端（未安装 qdrant-client 时在首次调用抛 ImportError）。"""
        if self._client is None:
            from qdrant_client import QdrantClient  # 延迟导入
            if self.url:
                self._client = QdrantClient(url=self.url)
            elif self.local_path:
                self._client = QdrantClient(path=self.local_path)
            else:
                self._client = QdrantClient(":memory:")
        return self._client

    def _ensure_collection(self, dim: int) -> None:
        """集合不存在时按向量维度创建（COSINE 距离度量）。"""
        from qdrant_client.models import Distance, VectorParams
        if not self._qdrant.collection_exists(self.collection):
            self._qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _collect_chunks(self, dir_path: str) -> List[Dict[str, str]]:
        """读取目录下所有 .md/.txt 并分块，返回 [{content, source}]。"""
        path = Path(dir_path)
        chunks: List[Dict[str, str]] = []
        if not path.is_dir():
            return chunks
        for f in sorted(path.glob("*")):
            if f.suffix.lower() not in (".md", ".txt"):
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            for chunk in _split(text, self.chunk_size, self.chunk_overlap):
                if chunk.strip():
                    chunks.append({"content": chunk, "source": f.name})
        return chunks

    async def ingest_directory(self, dir_path: str, force: bool = False) -> int:
        """摄入目录文档：分块 → embedding → 写入 Qdrant，返回总块数。

        - force=True 时先删除旧集合重建（用于知识库变更后重建索引）
        - 幂等：集合已有数据则跳过，避免重复摄入
        """
        from qdrant_client.models import PointStruct

        client = self._qdrant
        if force and client.collection_exists(self.collection):
            client.delete_collection(self.collection)

        # 幂等判断
        if client.collection_exists(self.collection):
            try:
                if client.count(collection_name=self.collection, exact=True).count > 0:
                    return client.count(
                        collection_name=self.collection, exact=True
                    ).count
            except Exception:
                pass

        chunks = self._collect_chunks(dir_path)
        if not chunks:
            return 0

        # 批量 embedding（一次调用，减少往返）
        texts = [c["content"] for c in chunks]
        vectors = await self.embedder.embed_texts(texts)
        dim = len(vectors[0])
        self._ensure_collection(dim)

        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={"content": c["content"], "source": c["source"]},
            )
            for i, (c, vec) in enumerate(zip(chunks, vectors))
        ]
        client.upsert(collection_name=self.collection, points=points)
        return len(points)

    @property
    def chunk_count(self) -> int:
        """已索引的文档块数量（与 TFIDFRetriever.chunk_count 对称）。"""
        try:
            return self._qdrant.count(
                collection_name=self.collection, exact=True
            ).count
        except Exception:
            return 0

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """embedding 查询向量 → Qdrant 相似度检索 → 返回 [{content, score, source}]。"""
        client = self._qdrant
        if not client.collection_exists(self.collection):
            return []
        qvec = await self.embedder.embed_query(query)
        # qdrant-client >=1.10 已移除 search()，改用 query_points()；参数 query_vector -> query
        hits = client.query_points(
            collection_name=self.collection,
            query=qvec,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "content": h.payload["content"],
                "score": round(h.score, 4),
                "source": h.payload.get("source"),
            }
            for h in hits
        ]

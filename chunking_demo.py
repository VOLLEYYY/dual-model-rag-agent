"""分块策略演示 — 简单文档 vs 复杂文档

一个独立可运行的脚本，演示两种「文档切片（分块）」策略：

1. 简单文档 → 定长分块
   - 适合：结构简单、段落短小的纯文本
   - 做法：定长 512 字符切开，相邻块重叠 50 字符
   - 优点：简单、确定性、零依赖；缺点：可能把语义单元切碎

2. 复杂文档 → 语义 / 结构分块
   - 适合：有 Markdown 标题、段落层级、代码块混排的长文档
   - 做法：先按标题(#/##)切 section → 再按空行切段落 → 段内超长再按句子合并
     （可选）传入 embedder 后，在句子边界找「相似度骤降点」断开，实现真正的语义分段

判定规则（classify_document）：满足任一 → 复杂文档
  ① 含 Markdown 标题
  ② 含代码块围栏 ```
  ③ 存在单段超过 chunk_size 的超长段落

运行：
    python chunking_demo.py          # 默认：简单→定长，复杂→结构分块
    # 想启用真正的「语义分段」，在 main() 里传入 embedder 即可（见文件底部注释）
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

# ============================================================
#  共享参数
# ============================================================
CHUNK_SIZE = 512   # 目标块大小（字符）
OVERLAP = 50       # 定长分块的重叠（字符）


# ============================================================
#  1. 定长分块（简单文档）
# ============================================================
def split_fixed(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> List[str]:
    """定长字符分块，overlap 防止关键信息被切在边界。

    与项目 rag.py 里的 _split() 逻辑一致。
    """
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
#  2. 简单 / 复杂文档判定
# ============================================================
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def classify_document(text: str) -> str:
    """判定文档复杂度，返回 "complex" 或 "simple"。

    判据（满足任一即视为复杂文档）：
      ① 含 Markdown 标题（# / ## / ...）
      ② 含代码块围栏（```）
      ③ 存在单段超过 chunk_size 的超长段落
    """
    if _HEADING_RE.search(text):
        return "complex"
    if _CODE_FENCE_RE.search(text):
        return "complex"
    for para in text.split("\n\n"):
        if len(para.strip()) > CHUNK_SIZE:
            return "complex"
    return "simple"


# ============================================================
#  3. 结构分块（复杂文档核心）
# ============================================================
def split_by_headings(text: str) -> List[Tuple[str, str]]:
    """按 Markdown 标题切分，返回 [(标题, 正文)]。

    每个 # 标题开启一个新 section；标题之前的正文归入 "(前言)"。
    """
    sections: List[Tuple[str, str]] = []
    cur_title = "(前言)"
    cur_body: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if cur_body:
                sections.append((cur_title, "\n".join(cur_body).strip()))
            cur_title = f"{m.group(1)} {m.group(2).strip()}"
            cur_body = []
        else:
            cur_body.append(line)
    if cur_body:
        sections.append((cur_title, "\n".join(cur_body).strip()))
    return sections


def split_by_paragraph(text: str) -> List[str]:
    """按空行分段落。"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# ============================================================
#  4. 语义分段（可选增强）
# ============================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")


def split_by_sentence(text: str) -> List[str]:
    """按句子边界切开；代码块（``` 围栏）整体保留，不拆散。"""
    sentences: List[str] = []
    in_code = False
    code_buf: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:  # 进入代码块
                in_code = True
                code_buf = [line]
            else:            # 结束代码块，整块作为一个「句子」
                code_buf.append(line)
                sentences.append("\n".join(code_buf))
                in_code = False
                code_buf = []
            continue
        if in_code:          # 代码块内部：整行保留
            code_buf.append(line)
            continue
        for seg in _SENT_SPLIT_RE.split(stripped):
            seg = seg.strip()
            if seg:
                sentences.append(seg)
    return sentences


def _cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def split_semantic(
    text: str,
    embedder: Optional[Callable[[str], List[float]]] = None,
    max_len: int = CHUNK_SIZE,
    threshold: float = 0.8,
) -> List[str]:
    """语义分段：优先在「语义骤降点」断开，否则按句子贪心合并到 max_len。

    - 传了 embedder：逐句算相邻句余弦相似度，相似度 < threshold 说明话题切换，断开。
    - 没传 embedder：退化为「按句子边界贪心合并到 max_len」，仍比定长分块更贴合语义。

    embedder 示例（需 sentence-transformers）：
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        embedder = lambda t: model.encode(t, normalize_embeddings=True).tolist()
    """
    sentences = split_by_sentence(text)
    if not sentences:
        return []

    # 有 embedder 时，计算相邻句相似度，找语义边界
    boundaries = None
    if embedder is not None:
        vecs = [embedder(s) for s in sentences]
        boundaries = {0}
        for i in range(len(sentences) - 1):
            if _cosine(vecs[i], vecs[i + 1]) < threshold:
                boundaries.add(i + 1)

    # 贪心合并：在语义边界处断开，且每块累计不超过 max_len
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0
    for i, s in enumerate(sentences):
        # 语义边界（仅当传了 embedder 时生效；否则只在超长时断开）
        if boundaries is not None and i in boundaries and buf:
            chunks.append("".join(buf))
            buf = []
            cur_len = 0
        # 超长时强制断开，避免单块过大
        if buf and cur_len + len(s) > max_len:
            chunks.append("".join(buf))
            buf = []
            cur_len = 0
        buf.append(s)
        cur_len += len(s)
    if buf:
        chunks.append("".join(buf))
    return [c for c in chunks if c.strip()]


# ============================================================
#  5. 统一入口：按文档复杂度派发
# ============================================================
def chunk_document(
    text: str,
    embedder: Optional[Callable[[str], List[float]]] = None,
) -> Tuple[str, List[str]]:
    """统一的文档切片入口。

    返回 (mode, chunks)：
      - mode = "fixed"      简单文档 → 定长分块
      - mode = "structural" 复杂文档 → 结构分块（标题→段落→句子/定长兜底）
      - mode = "semantic"   复杂文档 + 提供 embedder → 语义分段
    """
    mode = classify_document(text)

    if mode == "simple":
        return "fixed", split_fixed(text)

    # 复杂文档：先按标题结构切，段落过长时再按句子/语义细化
    chunks: List[str] = []
    for title, body in split_by_headings(text):
        if not body:
            continue
        if len(body) <= CHUNK_SIZE:
            chunks.append(body if title == "(前言)" else f"{title}\n{body}")
        else:
            sub = split_semantic(body, embedder=embedder)
            for i, s in enumerate(sub):
                prefix = "" if (title == "(前言)" or i > 0) else f"{title}\n"
                chunks.append(prefix + s)

    return ("semantic" if embedder is not None else "structural"), chunks


# ============================================================
#  演示
# ============================================================
_SIMPLE_DOC = """RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写，它把外部知识库的检索能力接入大语言模型，让模型在回答时能够引用真实的资料。

大模型本身只靠训练时的记忆回答问题，容易产生幻觉，也就是一本正经地编造不存在的内容。RAG 的做法是：先把问题拿去知识库里检索最相关的片段，再把这些片段和问题一起喂给模型，让它基于资料回答。

这样做的好处是，回答有了依据、可以标注来源，而且知识库更新之后无需重新训练模型，维护成本很低。对于企业内部知识库、产品手册这类内容频繁更新的场景，RAG 尤其合适。

在本项目里，简单文档直接用定长分块处理，把长文本切成固定大小的片段，相邻片段之间保留少量重叠，避免关键信息正好被切在边界上。

不过定长分块也有局限：它完全不看句子和段落的边界，可能把一个完整的句子拦腰截断，影响后续检索和生成的质量。所以它更适合段落短小、结构扁平的简单文档。

实际使用中，定长分块的参数需要权衡：块太小会丢失上下文，块太大又会让检索不够聚焦，同时检索速度也会下降。本项目默认取 512 字符、50 字符重叠，是一个经验上的折中值。你也可以根据具体场景微调这两个参数，例如问答类知识库用更小的块、长文档用更大的块。"""

_COMPLEX_DOC = """# RAG 分块策略

本文介绍 RAG 系统里对知识库文档做分块（切片）的几种策略，以及如何根据文档复杂度选择合适的方案。

## 1. 定长分块

定长分块是最基础的做法：按固定字符数把文本切开，相邻块之间保留一定重叠。它实现简单、结果确定，缺点是会把句子甚至单词切碎，破坏语义完整性。

## 2. 语义分块

语义分块的思路是让每个块都尽可能是一个完整的语义单元，避免把一句完整的话切成两半。常见做法有两种：第一种是按文档结构切分，例如利用 Markdown 的标题、段落、列表等天然边界，把内容组织成层次分明的块；第二种是用 embedding 模型计算相邻句子的相似度，在相似度骤降的位置断开，因为相似度骤降通常意味着话题发生了切换。相比定长分块，语义分块能更好地保持块的语义完整，检索时命中更精准，但实现成本也更高，往往需要额外的 embedding 计算。此外，语义分块还可以和标题结构结合起来使用，先按标题粗切、再在段落内部按语义边界细切，从而同时兼顾结构化信息与语义完整性，这正是复杂文档最合适的处理方式。

下面是一个示意代码块：

```python
def classify_document(text: str) -> str:
    if has_heading(text) or has_code_fence(text):
        return "complex"
    return "simple"
```

因此对于一个既有标题结构、又包含超长段落和代码块的复杂文档，应当先按标题切分，段落过长的再按句子或语义边界细化，而不是一刀切地按固定长度处理。"""


def _show(mode: str, chunks: List[str]) -> None:
    print(f"切片模式：{mode}    共 {len(chunks)} 块\n" + "-" * 60)
    for i, c in enumerate(chunks, 1):
        preview = c.replace("\n", " ⏎ ")[:72]
        print(f"[{i:02d}] {len(c):3d} 字符 | {preview}…")
    print()


def main() -> None:
    print("=" * 60)
    print("【示例 1】简单文档")
    print("=" * 60)
    print(f"复杂度判定：{classify_document(_SIMPLE_DOC)}")
    _show(*chunk_document(_SIMPLE_DOC))

    print("=" * 60)
    print("【示例 2】复杂文档")
    print("=" * 60)
    print(f"复杂度判定：{classify_document(_COMPLEX_DOC)}")
    _show(*chunk_document(_COMPLEX_DOC))

    print("提示：想启用真正的「语义分段」，给 chunk_document 传入 embedder 即可，")
    print("例如：")
    print("    from sentence_transformers import SentenceTransformer")
    print("    m = SentenceTransformer('BAAI/bge-small-zh-v1.5')")
    print("    mode, chunks = chunk_document(doc, embedder=lambda t: m.encode(t).tolist())")


if __name__ == "__main__":
    main()

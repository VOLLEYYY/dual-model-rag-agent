"""Gradio 聊天界面 — 双模型校验知识库 Agent 的前端展示层（Day 16）

运行：
    1. 先启动后端：python run.py（或 docker compose up -d agent-api）
    2. 再启动界面：python -m app.ui.chat
    3. 浏览器打开 http://localhost:7860

后端地址可用环境变量 BACKEND_URL 覆盖，默认 http://127.0.0.1:8000/api/ask。
"""
from __future__ import annotations

import json
import os

import gradio as gr
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/api/ask")
# 重建索引端点：由 /api/ask 推导出 /api/knowledge/reindex
REINDEX_URL = BACKEND_URL.rsplit("/ask", 1)[0] + "/knowledge/reindex"
# 流式问答端点：由 /api/ask 推导出 /api/ask/stream（SSE）
STREAM_URL = BACKEND_URL + "/stream"


def ask(question: str, use_dual_model: bool, top_k: float) -> tuple[str, str]:
    """调用后端非流式 /api/ask，返回 (回答, 评审详情)。

    说明：这是非流式实现（一次请求拿全量结果）。界面默认已改用 ask_stream
    走 /api/ask/stream 流式接口；保留本函数作为非流式备选/对比。
    """
    question = (question or "").strip()
    if not question:
        return "请输入问题。", ""

    try:
        resp = httpx.post(
            BACKEND_URL,
            json={
                "question": question,
                "use_dual_model": bool(use_dual_model),
                "top_k": int(top_k),
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"❌ 请求后端失败：{e}", ""

    answer = data.get("answer") or "（空回答）"

    details: list[str] = []
    if data.get("review_score") is not None:
        details.append(f"模型B 评分：{data['review_score']}/10")
    if data.get("need_revision"):
        details.append("模型B 已触发修订")
    if data.get("review_issues"):
        details.append("评审问题：" + "；".join(str(i) for i in data["review_issues"]))
    if data.get("safety_passed") is False:
        details.append("⚠️ 安全检查未通过")
    sources = data.get("sources") or []
    names = [s.get("source") for s in sources if s.get("source")]
    if names:
        details.append("来源：" + "、".join(names))

    return answer, "\n".join(details)


async def ask_stream(question: str, use_dual_model: bool, top_k: float):
    """调用后端 /api/ask/stream（SSE），流式返回 (回答, 评审详情)。

    逐 token 更新「回答」框，让用户先看到模型A 生成的内容；
    模型B 校验在后台完成后，评审详情一次性填充到「评审详情」框。
    事件序列见 agent_pipeline.run_pipeline_stream 的 docstring。
    """
    question = (question or "").strip()
    if not question:
        yield "请输入问题。", ""
        return

    payload = {
        "question": question,
        "use_dual_model": bool(use_dual_model),
        "top_k": int(top_k),
    }

    partial: list[str] = []        # 模型A 逐 token 累积
    review: dict = {}             # review 事件数据（评分/问题/是否修订）
    sources: list[str] = []       # 来源文件名
    safety_safe: bool = True

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", STREAM_URL, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("type")
                    if etype == "token":
                        partial.append(evt.get("content", ""))
                        yield "".join(partial), ""
                    elif etype == "retrieved":
                        sources = [s for s in evt.get("sources", []) if s]
                    elif etype == "safety":
                        safety_safe = bool(evt.get("safe", True))
                    elif etype == "review":
                        review = evt
                    elif etype == "done":
                        # done.answer 覆盖 token 累积（模型B 修订版优先）
                        if evt.get("answer"):
                            partial = [evt["answer"]]
                        sources = [s for s in evt.get("sources", sources) if s]
                        safety_safe = bool(evt.get("safety_passed", safety_safe))
    except Exception as e:
        if partial:
            yield "".join(partial) + f"\n\n⚠️ 流式响应中断：{e}", ""
        else:
            yield f"❌ 请求后端失败：{e}", ""
        return

    yield "".join(partial), _format_details(review, safety_safe, sources)


def _format_details(review: dict, safety_safe: bool, sources: list[str]) -> str:
    """把 review / safety / sources 组装成评审详情文本（与 ask 的细节字段一致）。"""
    details: list[str] = []
    if review.get("score") is not None:
        details.append(f"模型B 评分：{review['score']}/10")
    elif review.get("comment"):
        # 模型B 校验失败降级时，score 为空，展示 comment 说明
        details.append(review["comment"])
    if review.get("need_revision"):
        details.append("模型B 已触发修订")
    if review.get("issues"):
        details.append("评审问题：" + "；".join(str(i) for i in review["issues"]))
    if not safety_safe:
        details.append("⚠️ 安全检查未通过")
    if sources:
        details.append("来源：" + "、".join(sources))
    return "\n".join(details)


def reindex() -> str:
    """调用后端 /api/knowledge/reindex 重建索引，返回结果文本。"""
    try:
        resp = httpx.post(REINDEX_URL, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"❌ 重建索引失败：{e}"

    lines = [data.get("message", "重建完成")]
    docs = data.get("documents") or []
    if docs:
        lines.append("文档清单：" + "、".join(docs))
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面（问答 + 双模型开关 + 评审详情 + 知识库重建）。"""
    with gr.Blocks(title="双模型校验知识库 Agent") as demo:
        gr.Markdown("# 🤖 双模型校验知识库 Agent")
        gr.Markdown("RAG 检索 → 模型A 生成 → 安全检查 → 模型B 校验（双模型制衡）")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="问题",
                    placeholder="例如：什么是 RAG？",
                    lines=2,
                )
                with gr.Row():
                    use_dual = gr.Checkbox(label="启用双模型校验", value=True)
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="检索文档数 top_k",
                    )
                submit = gr.Button("提问", variant="primary")
            with gr.Column(scale=2):
                answer = gr.Textbox(label="回答", lines=10, interactive=False)
                detail = gr.Textbox(label="评审详情", lines=4, interactive=False)

        submit.click(
            ask_stream,
            inputs=[question, use_dual, top_k],
            outputs=[answer, detail],
        )
        question.submit(
            ask_stream,
            inputs=[question, use_dual, top_k],
            outputs=[answer, detail],
        )

        with gr.Accordion("📚 知识库管理", open=False):
            gr.Markdown(
                "把 `.md` / `.txt` 文档放进 `data/knowledge_base/` 后，"
                "点下面按钮重建索引（新文档才会被检索到）。"
            )
            reindex_btn = gr.Button("重建知识库索引", variant="secondary")
            reindex_result = gr.Textbox(label="结果", lines=3, interactive=False)

        reindex_btn.click(reindex, outputs=[reindex_result])

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        footer_links=["api", "settings"],
    )

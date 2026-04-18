"""Gradio frontend for RAG system."""
import re
import uuid
from typing import List, Dict

import gradio as gr

from app.rag_pipeline import get_pipeline
from app.embeddings import get_embedding_stats
from app.agents import agent_workflow, get_agent_chat_stats

def normalize_latex_delimiters(text: str) -> str:
    """Normalize common LaTeX delimiters so Gradio renders math reliably."""
    normalized = text.strip()
    normalized = re.sub(r"\\\((.*?)\\\)", r"$\1$", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", normalized, flags=re.DOTALL)
    return normalized


_ERROR_MESSAGES = {
    "NEWCONV": "You have reached the context limit for this conversation. Please start a new one.",
    "NORES":   "An error occurred while processing your question. Please try again.",
    "RATE":    "The rate limit has been exceeded. Please wait a moment and try again.",
    "EX":      "An error occurred while searching the knowledge base. Please try again later.",
}


def create_interface():
    """Create Gradio interface for RAG system."""

    pipeline = get_pipeline()
    pipeline.load_index()

    def _make_conv_id() -> str:
        return str(uuid.uuid4())[:8]

    def _conv_choices(store: dict):
        """Return display labels for gr.Radio, newest first."""
        return [
            (v["title"], k)
            for k, v in reversed(list(store.items()))
        ]

    def _active_history(store: dict, active_id: str) -> list:
        if not active_id or active_id not in store:
            return []
        return store[active_id]["history"]

    def create_conversation(store: dict):
        cid = _make_conv_id()
        store = dict(store)
        store[cid] = {"title": "New conversation", "history": []}
        choices = _conv_choices(store)

        return (
            store,          
            cid,        
            gr.Radio(choices=choices, value=cid), 
            [],             
            [],             
            "",             
            "",             
        )

    def select_conversation(selected_id: str, store: dict):
        if not selected_id:
            return store, "", [], [], "", "", ""

        history = _active_history(store, selected_id)
        return store, selected_id, history, [], "", "", ""

    def answer_question(
        question: str,
        store: dict,
        active_id: str,
        agent_mode: str,
    ):
        # Map display labels to internal mode keys
        _mode_map = {
            "Internal (RAG)": "internal",
            "External (Web)": "external",
            "Synthesized (Both)": "synthesized",
        }
        mode = _mode_map.get(agent_mode, "synthesized")

        if not active_id or active_id not in store:
            gr.Warning("Please create or select a conversation first.")
            return store, gr.skip(), active_id, _active_history(store, active_id), [], "", "", question

        if not question.strip():
            return store, gr.skip(), active_id, _active_history(store, active_id), [], "", "Please enter a question", question

        history = _active_history(store, active_id)
        result = agent_workflow(question, history, mode=mode)

        if str(result["answer"]).startswith("ER"):
            error_code = result["answer"][2:]
            error_msg  = _ERROR_MESSAGES.get(error_code, "An unknown error occurred.")
            new_history = history + [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": error_msg},
            ]
            store = dict(store)
            store[active_id]["history"] = new_history
            return store, gr.skip(), active_id, new_history, [], "", "", ""

        if result["sources"]:
            sources_data = []
            for source in result["sources"][:8]:
                origin = source.get("origin", "internal")
                if origin == "external":
                    sources_data.append([
                        source.get("source", "Web"),
                        source.get("url", ""),
                        "Web",
                        source.get("snippet", "")[:100],
                    ])
                else:
                    sources_data.append([
                        source['source'],
                        f"Page {int(source.get('page', 0))}",
                        f"{source.get('relevance', 0):.1%}",
                        source.get('preview', '')[:100],
                    ])
        else:
            sources_data = [["No sources", "", "N/A", "No relevant sources found"]]

        embed_stats = get_embedding_stats()
        agent_stats = get_agent_chat_stats()
        chat_stats  = pipeline.get_chat_stats()
        # Merge: pipeline tracks its own calls, agent tracks agent calls
        chat_stats = {
            "input_tokens": chat_stats["input_tokens"] + agent_stats["input_tokens"],
            "output_tokens": chat_stats["output_tokens"] + agent_stats["output_tokens"],
            "total_cost": chat_stats["total_cost"] + agent_stats["total_cost"],
        }

        judge_scores = result.get("judge_scores", {})
        scores_text = ""
        if judge_scores:
            accuracy = judge_scores.get('accuracy', 0)
            hallucination = judge_scores.get('hallucination', 0)
            relevance = judge_scores.get('relevance', 0)
            confidence = judge_scores.get('confidence', 0)

            scores_text = (
                f"### Quality Assessment\n\n"
                f"**Overall Confidence:** {confidence}/100\n\n"
                f"| Metric | Score |\n"
                f"|--------|-------|\n"
                f"| Accuracy | {accuracy}/100 |\n"
                f"| Hallucination-Free | {hallucination}/100 |\n"
                f"| Relevance | {relevance}/100 |"
            )
        else:
            scores_text = "Quality scores pending..."
        judge_text = ""
        if judge_scores:
            judge_text = (
                f"\n\n**Quality Score (LLM Judge):**\n"
                f"- Accuracy: {judge_scores.get('accuracy', 'N/A')}/100\n"
                f"- Hallucination-Free: {judge_scores.get('hallucination', 'N/A')}/100\n"
                f"- Relevance: {judge_scores.get('relevance', 'N/A')}/100\n"
                f"- Overall Confidence: {judge_scores.get('confidence', 'N/A')}/100"
            )

        stats_text = (
            f"**Cost Tracking:**\n"
            f"- Total cost so far: ${embed_stats['total_cost'] + chat_stats['total_cost']:.6f}\n\n"
            f"**Token Usage:**\n"
            f"- Embeddings: {embed_stats['total_tokens']:,}\n"
            f"- Chat input: {chat_stats['input_tokens']:,}\n"
            f"- Chat output: {chat_stats['output_tokens']:,}"
            f"{judge_text}"
        )

        formatted_answer = normalize_latex_delimiters(result["answer"])

        new_history = history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": formatted_answer},
        ]

        store = dict(store)
        if store[active_id]["title"] == "New conversation":
            store[active_id]["title"] = question[:40] + ("…" if len(question) > 40 else "")
        store[active_id]["history"] = new_history

        choices = _conv_choices(store)
        updated_conv_list = gr.Radio(choices=choices, value=active_id)

        updated_conv_list = gr.update(choices=choices, value=active_id)

        return store, updated_conv_list, active_id, new_history, sources_data, scores_text, stats_text, ""


    custom_css = """
    """

    with gr.Blocks(title="RAG System for Math ML", css=custom_css) as demo:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>RAG System for Mathematics for Machine Learning</h1>", )

        conv_store  = gr.State({})
        active_id   = gr.State("")

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                agent_mode = gr.Radio(
                    label="Agent Mode",
                    choices=["Internal (RAG)", "External (Web)", "Synthesized (Both)"],
                    value="Synthesized (Both)",
                    interactive=True,
                )
                new_conv_btn = gr.Button("＋ New Conversation", variant="secondary")
                conv_list = gr.Radio(
                    label="Conversations",
                    choices=[],
                    value=None,
                    interactive=True,
                )

            with gr.Column(scale=3):
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask away…",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("Ask", size="lg", variant="primary", scale=1)

                chatbot = gr.Chatbot(
                    height=500,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$",  "right": "$",  "display": False},
                    ], 
                    show_label=False
                )

            with gr.Column(scale=1, min_width=200):
                gr.Textbox(
                    value=f"Index loaded: {len(pipeline.chunks)} chunks",
                    label="System Status",
                    interactive=False,
                )
                sources_output = gr.Dataframe(
                    headers=["Source", "Page / URL", "Relevance / Origin", "Preview"],
                    label="Sources (Top 8)",
                    interactive=False,
                )
                scores_output = gr.Markdown(label="Quality Scores", value="Scores will appear here…")
                stats_output   = gr.Textbox(label="Usage Stats", interactive=False, lines=6)


        new_conv_btn.click(
            fn=create_conversation,
            inputs=[conv_store],
            outputs=[conv_store, active_id, conv_list, chatbot, sources_output, scores_output, stats_output],
        )

        conv_list.select(
            fn=select_conversation,
            inputs=[conv_list, conv_store],
            outputs=[conv_store, active_id, chatbot, sources_output, scores_output, stats_output],
        )

        _ask_inputs   = [question_input, conv_store, active_id, agent_mode]
        _ask_outputs  = [conv_store, conv_list, active_id, chatbot, sources_output, scores_output, stats_output, question_input]

        submit_btn.click(fn=answer_question, inputs=_ask_inputs, outputs=_ask_outputs)
        question_input.submit(fn=answer_question, inputs=_ask_inputs, outputs=_ask_outputs)

        gr.Examples(
            examples=[
                ["What are the three concepts at the core of machine learning?"],
                ["Explain the trace of a matrix and its relationship to eigenvalues"],
                ["What is the Cholesky decomposition used for?"],
                ["Define the Taylor polynomial"],
                ["What are the roles of prior, likelihood, and posterior in Bayes theorem?"],
            ],
            inputs=question_input,
            label="Example Questions",
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

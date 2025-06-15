from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import aiml
import os
import time
import html


def init_aiml_kernel():
    kernel = aiml.Kernel()

    aiml_file = "user_driven.aiml"
    if not os.path.isfile(aiml_file):
        raise FileNotFoundError(f"AIML file not found: {aiml_file}")
    brain_file = "bot_brain.brn"
    if os.path.isfile(brain_file):
        kernel.bootstrap(brainFile=brain_file)
    else:
        kernel.learn(aiml_file)
        kernel.saveBrain(brain_file)

    return kernel


kernel = init_aiml_kernel()


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_100",
            body_background_fill_dark="*neutral_900",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="14px",  
        )


seafoam = Seafoam()


def preprocess_aiml_response(response: str) -> str:
    """处理 AIML 响应以在 Gradio 中正确显示"""
    response = response.replace("_br_", "\n\n")
    response = response.replace("_b_", "**")
    response = response.replace("_i_", "*")
    response = html.unescape(response)
    return response or "I'm not sure how to answer that. Try asking about careers, majors, or career preparation tips."


def respond(message: str, chat_history: list) -> tuple:
    processed_input = message.strip().upper()
    response = kernel.respond(processed_input)
    formatted_response = preprocess_aiml_response(response)
    time.sleep(0.3)
    chat_history.append((message, formatted_response))
    return chat_history, ""


with gr.Blocks(theme=seafoam) as demo:
    # 加标题和描述
    gr.Markdown("## Xplore Career Chatbot")
    gr.Markdown("This is **Xplore Career Chatbot**. You can ask questions about careers. Start by typing `hello` or `hi`.")

    chatbot = gr.Chatbot(label="Xplore Career Bot")

    with gr.Row():
        user_input = gr.Textbox(placeholder="Welcome to ask questions about career!", show_label=False)

    with gr.Row(equal_height=True):
        submit = gr.Button("Send", size="sm")
        clear = gr.Button("Clear", size="sm")

    with gr.Row():
        gr.Markdown("**Examples:**")
        example1 = gr.Button("CAREERS FOR AIT", size="sm")
        example2 = gr.Button("DESCRIBE AUDITOR", size="sm")
        example3 = gr.Button("CV HELP", size="sm")

    submit.click(respond, [user_input, chatbot], [chatbot, user_input])
    clear.click(lambda: [], None, chatbot)

    example1.click(lambda: "CAREERS FOR AIT", None, user_input)
    example2.click(lambda: "DESCRIBE AUDITOR", None, user_input)
    example3.click(lambda: "CV HELP", None, user_input)


demo.launch()


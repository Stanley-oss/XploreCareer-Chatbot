import gradio as gr
import io
from typing import Iterable
from chatbot import CombinedChatbot
from career_predictor import CareerPredictor
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def respond(message: str, chat_history: list) -> tuple:
    # Get bot response and append to history as dicts
    bot_response = combined_chatbot_instance.get_response(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_response})
    return chat_history, ""  # clear input box


def plot_svg(probs: dict) -> str:
    """
    把预测结果绘制成柱状图，返回SVG格式的字符串
    dict: {career : probability}
    """
    labels = list(probs.keys())
    values = list(probs.values())

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
    ax.set_facecolor('none')

    colors = plt.cm.tab10.colors # 使用matplotlib的tab10颜色，要改颜色这里改
    bar_colors = [colors[i % len(colors)] for i in range(len(values))]

    ax.barh(labels[::-1], values[::-1], color=bar_colors) # 水平柱状图

    max_val = max(values) if values else 1
    ax.set_xlim(0, max_val + 0.1)
    ax.set_xlabel("Probability")
    ax.set_title("Career Recommendation Probabilities")

    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    # ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7) # 加y轴网格线

    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format='svg', transparent=True)
    svg_data = buf.getvalue()
    buf.close()
    plt.close(fig)
    return svg_data

def update_chart(message: str) -> str:
    try:
        probs = predictor.predict(message)
    except NameError:
        raise RuntimeError("predict 函数未定义，请实现 predict(message) -> dict。")
    svg = plot_svg(probs)
    svg = svg.replace( # 注入SVG头居中显示
        "<svg ",
        '<svg height="100%" preserveAspectRatio="xMidYMid meet" '
    )
    html = (
        '<div style="background: transparent; width:100%; height:100%; '
        'display: flex; justify-content: center; align-items: center;">'
        f'{svg}'
        '</div>'
    )
    return html


if __name__ == "__main__":
    predictor = CareerPredictor()
    combined_chatbot_instance = CombinedChatbot()
    with gr.Blocks(theme=Seafoam()) as app:
        gr.Markdown("## Xplore Career Chatbot")
        gr.Markdown("This is **Xplore Career Chatbot**. You can ask questions about careers. Start by typing `hello` or `hi`.")
        
        # 聊天窗和概率图的比例为3:2
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Xplore Career Bot", type='messages')
            with gr.Column(scale=2):
                chart = gr.HTML(elem_id="prob_chart",)

        with gr.Row():
            user_input = gr.Textbox(placeholder="Welcome to ask questions about career!", show_label=False)

        with gr.Row(equal_height=True):
            submit = gr.Button("Send", size="sm")
            clear = gr.Button("Clear", size="sm")

        with gr.Row():
            gr.Markdown("**Examples:**")
            example1 = gr.Button("CAREERS FOR AIT", size="sm")
            example2 = gr.Button("START PLANNING", size="sm")
            example3 = gr.Button("CV HELP", size="sm")
        
        example1.click(lambda: "CAREERS FOR AIT", None, user_input)
        example2.click(lambda: "START PLANNING", None, user_input)
        example3.click(lambda: "CV HELP", None, user_input)

        submit.click(respond, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
        submit.click(update_chart, inputs=user_input, outputs=chart)

        clear.click(lambda: [], None, chatbot)

        def initial_load():
            combined_chatbot_instance.reset_all()
            welcome = combined_chatbot_instance.get_response("hello")
            return [{"role": "assistant", "content": welcome}]

        app.load(initial_load, None, chatbot)

    app.launch()
    print("INFO: Xplore Career Chatbot Launched.")

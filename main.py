import gradio as gr
import io
from typing import Iterable
from chatbot import Bot
from career_predictor import CareerPredictor
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from llm import LLMClient
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


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


async def respond(message: str, chat_history: list):
    """
    流式响应:
    获取原始Chatbot回复
    调用llm,获取改进后的回复
    """
    global user_response
    user_response = user_response + ' ' + message
    bot_response_original = combined_chatbot.get_response(message)
    chat_history.append({"role": "user", "content": message})
    ollama_input = f"User Input: {message}\nTemplate Response: {bot_response_original}\n"
    print(f"INFO: Chatbot Input: {bot_response_original}")

    # 流式获取改进后的响应
    improved_response = ""
    async for partial_response in llm_client.call_stream(ollama_input):
        improved_response = partial_response
        current_history = chat_history.copy()
        current_history.append({"role": "assistant", "content": improved_response})
        yield current_history, "", gr.update(visible=False), ""

    chat_history.append({"role": "assistant", "content": improved_response})
    yield chat_history, "", gr.update(visible=False), ""


def plot_svg(probs: dict) -> str:
    """
    把预测结果绘制成柱状图，返回SVG格式的字符串
    dict: {career : probability}
    """
    labels = list(probs.keys())
    values = list(probs.values())

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    ax.set_facecolor('none')

    colors = plt.cm.Set3.colors
    bar_colors = [colors[i % len(colors)] for i in range(len(values))]

    ax.barh(labels[::-1], values[::-1], color=bar_colors, alpha=0.8)

    max_val = max(values) if values else 1
    ax.set_xlim(0, max_val + 0.1)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_title("Career Recommendation Probabilities", fontsize=14, fontweight='bold')

    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format='svg', transparent=True, bbox_inches='tight')
    svg_data = buf.getvalue()
    buf.close()
    plt.close(fig)
    return svg_data


if __name__ == "__main__":
    user_response = ""
    predictor = CareerPredictor()
    combined_chatbot = Bot()

    system_prompt = "You are a professional Career Recommendation Bot by the name of Xplore Career Chatbot, dedicated to the career recommendation of Xiamen University Malaysia(XMUM) students. The following inputs are all user inputs with corresponding template responses, you need to give a lively, human-friendly and concise response based on the template responses. Your response better be framed by the template unless the template indicates that it does not know how to answer, then it will be you to answer the user. Do not insert links in your response, try to keep your response concise and clear. ATTENTION YOU ONLY NEED TO REPLY YOUR RESPONSE, DO NOT MENTION THE EXISTANCE OF THE TEMPLATE, YOU ARE DIRECTLY COMMUNICATING WITH THE USER."
    llm_client = LLMClient(system_prompt)

    with gr.Blocks(theme=Seafoam()) as app:
        gr.Markdown("## Xplore Career Chatbot")
        gr.Markdown(
            "This is **Xplore Career Chatbot**. You can ask questions about careers. Start by typing `hello` or `hi`.")

        chatbot = gr.Chatbot(label="Xplore Career Bot", type='messages', height=500)
        with gr.Row(equal_height=True):
            with gr.Column(scale=9):
                user_input = gr.Textbox(
                    placeholder="Welcome to ask questions about career! (Press Enter to send, Shift+Enter for new line)",
                    show_label=False
                )
            with gr.Column(scale=1):
                submit = gr.Button("Send", size="sm", variant="primary")

        with gr.Row():
            predict_btn = gr.Button("Predict", size="sm", variant="secondary")
            clear = gr.Button("Clear", size="sm")

        with gr.Column(visible=False) as prediction_panel:
            with gr.Row():
                gr.Markdown("### Career Prediction Results")
                close_prediction_btn = gr.Button("✕", size="sm", variant="secondary")
            prediction_chart = gr.HTML()

        with gr.Row():
            gr.Markdown("**Examples:**")
            example1 = gr.Button("CAREERS FOR AIT", size="sm")
            example2 = gr.Button("START PLANNING", size="sm")
            example3 = gr.Button("CV HELP", size="sm")

        example1.click(lambda: "CAREERS FOR AIT", None, user_input)
        example2.click(lambda: "START PLANNING", None, user_input)
        example3.click(lambda: "CV HELP", None, user_input)

        submit_event = submit.click(respond, inputs=[user_input, chatbot],
                                    outputs=[chatbot, user_input, prediction_panel, prediction_chart])

        user_input.submit(respond, inputs=[user_input, chatbot],
                          outputs=[chatbot, user_input, prediction_panel, prediction_chart])


        def show_prediction_panel():
            global user_response

            if not user_response.strip():
                return gr.update(visible=False), ""

            try:
                user_response = user_response.strip()
                print(f"INFO: Predicting user response: {user_response}")
                probs = predictor.predict(user_response)
                svg = plot_svg(probs)
                styled_chart = f"""
                <div style="
                    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(240,248,255,0.9));
                    border-radius: 15px;
                    padding: 25px;
                    margin: 10px 0;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    border: 1px solid rgba(255,255,255,0.2);
                    backdrop-filter: blur(10px);
                ">
                    {svg}
                </div>
                """
                return gr.update(visible=True), styled_chart
            except Exception as e:
                return gr.update(
                    visible=False), f"<div style='color: red; text-align: center; padding: 20px;'>Error: {str(e)}</div>"


        predict_btn.click(
            show_prediction_panel,
            inputs=[],
            outputs=[prediction_panel, prediction_chart]
        )

        close_prediction_btn.click(
            lambda: (gr.update(visible=False), ""),
            outputs=[prediction_panel, prediction_chart]
        )

        clear.click(lambda: [], None, chatbot)


        def initial_load():
            global user_response
            user_response = ""
            combined_chatbot.reset()
            welcome = combined_chatbot.get_response("hello")
            return [{"role": "assistant", "content": welcome}]


        app.load(initial_load, None, chatbot)

    app.launch()
    print("INFO: Xplore Career Chatbot Launched.")

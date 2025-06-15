import gradio as gr
import aiml
import os
import io
from contextlib import redirect_stdout
from typing import Iterable
from expert_system import UserProfile, inference_engine, RULE_BASE
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


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


class CareerChatbot:
    def __init__(self, aiml_file="career_dialogue.aiml"):
        # 加载 AIML 内核
        self.kernel = aiml.Kernel()
        if os.path.exists(aiml_file):
            self.kernel.learn(aiml_file)
        else:
            raise FileNotFoundError(f"AIML file '{aiml_file}' not found in the directory.")

        # 这是原来在 app.py 中的常量，现在移到这里
        self.CHALLENGE_MAP = {
            '1': 'dislikes group projects', '2': 'dislikes public speaking or presentations',
            '3': 'hard to come up with new, original ideas', '4': 'gets a headache from complex data or math',
            '5': 'prefers clear instructions over ambiguous tasks',
            '6': 'tends to lose the big picture when facing too much information',
            '7': 'not interested in business operations or how companies make profit',
            '8': 'gets anxious under pressure or tight deadlines',
            '9': 'finds it difficult to persuade others',
            '10': 'prefers to complete tasks independently rather than leading a team',
            '11': 'tends to procrastinate, deadlines are the main motivation',
            '12': 'afraid of or dislikes handling interpersonal conflicts',
            '13': 'gets bored easily by repetitive, routine tasks',
            '14': 'dislikes networking or actively building new connections',
            '15': 'afraid of making mistakes, tends to be a perfectionist',
            '16': 'finds it hard to maintain focus for long periods',
            '17': 'not good at reporting work to superiors or clients',
            '18': 'struggles with purely theoretical concepts, needs hands-on practice',
            '19': 'hesitates when making decisions'
        }


        self.formatted_challenge_list = """
1. dislikes group projects
2. dislikes public speaking
3. hard to come up with new ideas
4. dislikes complex data or math
5. prefers clear instructions
6. loses the big picture with too much info
7. not interested in business operations
8. anxious under pressure
9. finds it difficult to persuade others
10. prefers to work independently
11. tends to procrastinate
12. dislikes interpersonal conflicts
13. gets bored by routine tasks
14. dislikes networking
15. tends to be a perfectionist
16. finds it hard to maintain focus
17. not good at reporting work
18. struggles with purely theoretical concepts
19. hesitates when making decisions
"""

        # 初始化对话状态
        self.reset()

    def reset(self):

        self.conversation_state = 0
        self.user_data = {}
        print("INFO: Chatbot state has been reset.")

    def _generate_analysis_report(self, profile: UserProfile) -> str:

        f = io.StringIO()
        with redirect_stdout(f):
            inference_engine(profile, RULE_BASE)
            final_abilities = profile.abilities.sort_values(ascending=False)
            print("\n--- Final Ability Weights Analysis ---")
            for ability, score in final_abilities.items():
                print(f"{ability:<25} {score:+.2f}")
            print("=" * 35)
        return f.getvalue()

    def get_response(self, user_input: str) -> str:
        """这是UI唯一需要调用的函数，包含了完整的对话状态机逻辑。"""
        bot_response = ""
        user_input = user_input.strip()

        # (这里的状态机逻辑保持不变，从 if self.conversation_state == 0 ... 到 ... else: bot_response = "...")
        if self.conversation_state == 0:
            if user_input.lower() in ['hello', 'hi', '你好']:
                bot_response = self.kernel.respond("GREETING")
            elif user_input.lower() in ['start planning', 'start', 'yes', 'yeah', '开始', '好的']:
                bot_response = self.kernel.respond("STARTPLANNING")
                if bot_response:
                    self.conversation_state = 1
            else:
                bot_response = "Sorry, I didn't understand. Please say 'hello' or 'start'."
        elif self.conversation_state == 1:
            self.user_data['major'] = user_input
            template = self.kernel.respond("ASKINTERESTS")
            bot_response = template.format(major=self.user_data['major'])
            self.conversation_state = 2
        elif self.conversation_state == 2:
            self.user_data['interests'] = user_input
            template = self.kernel.respond("ASKMBTI")
            bot_response = template.format(interests=self.user_data['interests'])
            self.conversation_state = 3
        elif self.conversation_state == 3:
            if user_input.lower() in ["i don't know", "i dont know", "not sure", "不知道"]:
                self.user_data['mbti'] = "Unknown"
                template = self.kernel.respond("ASKCHALLENGESSKIPMBTI")
                bot_response = template
            else:
                self.user_data['mbti'] = user_input.upper()
                template = self.kernel.respond("ASKCHALLENGES")
                bot_response = template.format(mbti=self.user_data['mbti'])
            if template:
                self.conversation_state = 4
        elif self.conversation_state == 4:
            self.user_data['challenges_input'] = user_input
            template = self.kernel.respond("CONFIRMINFO")
            bot_response = template.format(**self.user_data)
            self.conversation_state = 5
        elif self.conversation_state == 5:
            if user_input.lower() in ['confirm', '确认']:
                final_message = self.kernel.respond("FINALRESPONSE")
                user_major = self.user_data.get('major', '')
                user_interests_str = self.user_data.get('interests', '')
                user_mbti = self.user_data.get('mbti', '')
                user_challenges_str = self.user_data.get('challenges_input', '')
                user_interests = [interest.strip() for interest in user_interests_str.split(',')]
                if user_mbti == 'Unknown': user_mbti = ""
                challenge_numbers = [num.strip() for num in user_challenges_str.split(',')]
                user_challenges = [self.CHALLENGE_MAP[num] for num in challenge_numbers if num in self.CHALLENGE_MAP]
                profile = UserProfile(major=user_major, interests=user_interests, mbti=user_mbti,
                                      challenges=user_challenges)
                analysis_result = self._generate_analysis_report(profile)
                bot_response = f"{final_message}\n\n```text\n{analysis_result}\n```\n\n[System] Analysis complete. You can say 'start over' to begin."
                self.reset()
                self.conversation_state = 6
            elif user_input.lower() in ['start over', '重新开始']:
                self.reset()
                bot_response = self.kernel.respond("STARTPLANNING")
                self.conversation_state = 1
            else:
                bot_response = "Sorry, I didn't understand. Please say 'confirm' or 'start over'."
        elif self.conversation_state == 6:
            if user_input.lower() in ['start over', '重新开始']:
                self.reset()
                bot_response = self.kernel.respond("STARTPLANNING")
                self.conversation_state = 1
            else:
                bot_response = "The analysis is complete. If you want to start a new one, please say 'start over'."
        if not bot_response:
            bot_response = "Sorry, an internal error occurred. Let's try starting over. What is your full major?"
            self.reset()
            self.conversation_state = 1


        if "[CHALLENGE_LIST]" in bot_response:
            bot_response = bot_response.replace("[CHALLENGE_LIST]", self.formatted_challenge_list)

        return bot_response


chatbot_instance = CareerChatbot()

seafoam = Seafoam()
with gr.Blocks(theme=seafoam) as demo:
    gr.Markdown("## Xplore Career Chatbot")
    gr.Markdown("This is **Xplore Career Chatbot**. Start your conversation below.")

    chatbot_ui = gr.Chatbot(label="Xplore Career Bot", height=500)

    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=4)
        submit_button = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_button = gr.Button("Clear History", variant="secondary")



    def respond(message, chat_history):
        bot_response = chatbot_instance.get_response(message)
        chat_history.append((message, bot_response))
        return "", chat_history



    def clear_and_reset():
        chatbot_instance.reset()
        return []


    def initial_load():
        chatbot_instance.reset()
        initial_message = chatbot_instance.get_response("hello")
        return [[None, initial_message]]


    submit_action = submit_button.click(respond, [user_input, chatbot_ui], [user_input, chatbot_ui], queue=False)
    user_input.submit(respond, [user_input, chatbot_ui], [user_input, chatbot_ui], queue=False)

    clear_button.click(clear_and_reset, None, chatbot_ui, queue=False).then(
        initial_load, None, chatbot_ui
    )

    demo.load(initial_load, None, chatbot_ui)

if __name__ == "__main__":
    demo.launch()
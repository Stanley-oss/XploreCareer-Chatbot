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


def init_aiml_kernel(aiml_file_name):
    kernel = aiml.Kernel()
    if not os.path.isfile(aiml_file_name):
        raise FileNotFoundError(f"AIML file not found: {aiml_file_name}")
    brain_file = "bot_brain.brn"
    if os.path.isfile(brain_file):
        kernel.bootstrap(brainFile=brain_file)
    else:
        kernel.learn(aiml_file_name)
        kernel.saveBrain(brain_file)
    return kernel


general_query_kernel = init_aiml_kernel("career_query.aiml")
guided_dialogue_kernel = init_aiml_kernel("career_dialogue.aiml")


#The connection part between the expert system and the robot-driven template based
class CareerChatbot:
    def __init__(self, aiml_file="career_dialogue.aiml"):
        self.kernel = aiml.Kernel()
        if os.path.exists(aiml_file):
            self.kernel.learn(aiml_file)
        else:
            raise FileNotFoundError(f"AIML file '{aiml_file}' not found in the directory.")

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

        #From the original connection part app.py, which is the only part of the UI that needs to be mobilized
        bot_response = ""
        user_input = user_input.strip()

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

        return bot_response


class CombinedChatbot:
    def __init__(self):
        self.guided_chatbot = CareerChatbot("career_dialogue.aiml")
        self.general_query_kernel = general_query_kernel
        self.mode = "general_query"  # Initial mode

    def process_aiml_formatting(self, text: str) -> str:
        text = text.replace('_br_', '\n\n')
        text = text.replace("_b_", "**")
        text = text.replace("_i_", "*")
        return text or "I'm not sure how to answer that. Try asking about careers, majors, or career preparation tips."

    def get_response(self, user_input: str) -> str:
        user_input_lower = user_input.strip().lower()

        # Handle mode switching commands
        if user_input_lower == "start planning":
            self.mode = "guided_planning"
            self.guided_chatbot.reset() # Ensure a fresh start for guided mode
            return self.process_aiml_formatting(self.guided_chatbot.get_response(user_input)) # Initial prompt for guided mode
        elif user_input_lower == "ask general":
            self.mode = "general_query"
            return self.process_aiml_formatting(self.general_query_kernel.respond("HELLO")) # Greet in general mode
        elif user_input_lower == "start over" and self.mode == "guided_planning":
            self.guided_chatbot.reset()
            return self.process_aiml_formatting(self.guided_chatbot.get_response(user_input)) # Restart guided
        elif user_input_lower == "cancel planning" and self.mode == "guided_planning":
            self.mode = "general_query"
            self.guided_chatbot.reset()
            return "Career planning cancelled. You are now in general query mode. Type 'start planning' to begin a new plan or 'hello' to ask general questions."


        if self.mode == "guided_planning":
            return self.process_aiml_formatting(self.guided_chatbot.get_response(user_input))
        else: # general_query mode
            response = self.general_query_kernel.respond(user_input.strip().upper())
            return self.process_aiml_formatting(response)

    def reset_all(self):
        self.guided_chatbot.reset()
        self.mode = "general_query" # Reset to default mode
        print("INFO: All chatbot states reset.")


def respond(message: str, chat_history: list) -> tuple:
    bot_response = combined_chatbot_instance.get_response(message)
    chat_history.append((message, bot_response))
    return chat_history, ""

# 1. Create a unique instance of chatbot
combined_chatbot_instance = CombinedChatbot()


# 2. Define the UI layout
seafoam = Seafoam()
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
        example2 = gr.Button("START PLANNING", size="sm")
        example3 = gr.Button("CV HELP", size="sm")

    submit.click(respond, [user_input, chatbot], [chatbot, user_input])
    clear.click(lambda: [], None, chatbot)

    example1.click(lambda: "CAREERS FOR AIT", None, user_input)
    example2.click(lambda: "START PLANNING", None, user_input)
    example3.click(lambda: "CV HELP", None, user_input)


    def initial_load_combined():
        combined_chatbot_instance.reset_all()
        initial_message = combined_chatbot_instance.get_response("hello")
        return [(None, initial_message)]

    demo.load(initial_load_combined, None, chatbot)

if __name__ == "__main__":
    demo.launch()

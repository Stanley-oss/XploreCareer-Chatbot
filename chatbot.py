import aiml
import os
import io
from contextlib import redirect_stdout
import requests

from expert_system import UserProfile, inference_engine, RULE_BASE


# 尝试把careerchatbot和combinedchatbot合并到一起，不然逻辑太乱了
class Bot:
    def __init__(self):
        self.kernel = aiml.Kernel()
        aiml_files = ["career_query.aiml", "career_dialogue.aiml"]
        for file in aiml_files:
            if os.path.exists(file):
                self.kernel.learn(file)

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

    def process_aiml_formatting(self, text: str) -> str:
        text = text.replace('_br_', '\n\n')
        text = text.replace("_b_", "**")
        text = text.replace("_i_", "*")
        return text or "I'm not sure how to answer that. Try asking about careers, majors, or career preparation tips."

    def get_response(self, user_input: str) -> str:
        user_input = user_input.strip().lower()

        # 初始状态默认是general
        if self.conversation_state == 0:
            if user_input == "start planning":
                self.conversation_state = 1
                return self.process_aiml_formatting(self.kernel.respond("STARTPLANNING"))
            else:
                return self.process_aiml_formatting(self.kernel.respond(user_input.upper()))

        # 处理中途退出
        if user_input == "start over":
            self.reset()
            return self.process_aiml_formatting(self.kernel.respond("STARTPLANNING"))
        if user_input == "cancel planning":
            self.reset()
            return "Career planning cancelled. You can now ask general questions or start a new plan."
        if user_input == "ask general":
            return self.process_aiml_formatting(self.kernel.respond("HELLO"))

        # 生成plan的部分
        if self.conversation_state == 1:
            self.user_data['major'] = user_input
            self.conversation_state = 2
            template = self.kernel.respond("ASKINTERESTS")
            formatted = template.format(major=self.user_data['major'])
            return self.process_aiml_formatting(formatted)

        if self.conversation_state == 2:
            self.user_data['interests'] = user_input
            self.conversation_state = 3
            template = self.kernel.respond("ASKMBTI")
            formatted = template.format(interests=self.user_data['interests'])
            return self.process_aiml_formatting(formatted)

        if self.conversation_state == 3:
            if user_input in ["i don't know", "i dont know", "not sure", "不知道"]:
                self.user_data['mbti'] = "Unknown"
                self.conversation_state = 4
                return self.process_aiml_formatting(self.kernel.respond("ASKCHALLENGESSKIPMBTI"))
            else:
                self.user_data['mbti'] = user_input.upper()
                self.conversation_state = 4
                template = self.kernel.respond("ASKCHALLENGES")
                formatted = template.format(mbti=self.user_data['mbti'])
                return self.process_aiml_formatting(formatted)

        if self.conversation_state == 4:
            self.user_data['challenges_input'] = user_input
            self.conversation_state = 5
            template = self.kernel.respond("CONFIRMINFO")
            formatted = template.format(**self.user_data)
            return self.process_aiml_formatting(formatted)

        if self.conversation_state == 5:
            if user_input in ["confirm", "确认"]:
                profile = self._build_user_profile()
                analysis_result = self._generate_analysis_report(profile)
                final_message = self.kernel.respond("FINALRESPONSE")
                self.reset()
                self.conversation_state = 6  # technically reset already,保留6状态用于明确结束
                return f"{final_message}\n\n```text\n{analysis_result}\n```\n\n[System] Analysis complete. You can say 'start over' to begin."
            elif user_input == "start over":
                self.reset()
                self.conversation_state = 1
                return self.process_aiml_formatting(self.kernel.respond("STARTPLANNING"))
            else:
                return "Please say 'confirm' to complete or 'start over' to restart."

        if self.conversation_state == 6:
            return "The analysis is complete. If you want to start a new one, please say 'start over'."

        return "Unexpected error. Restarting...\n" + self.kernel.respond("STARTPLANNING")

    def _build_user_profile(self) -> UserProfile:
        major = self.user_data.get('major', '')
        interests = [i.strip() for i in self.user_data.get('interests', '').split(',')]
        mbti = self.user_data.get('mbti', '')
        if mbti == "Unknown":
            mbti = ""
        challenges_str = self.user_data.get('challenges_input', '')
        challenge_numbers = [num.strip() for num in challenges_str.split(',')]
        challenges = [self.CHALLENGE_MAP[num] for num in challenge_numbers if num in self.CHALLENGE_MAP]
        return UserProfile(major=major, interests=interests, mbti=mbti, challenges=challenges)

import aiml
import os
import io
from contextlib import redirect_stdout
import requests

from expert_system import UserProfile, inference_engine, RULE_BASE

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

        bot_response = ""
        user_input = user_input.strip()

        if self.conversation_state == 0:
            if user_input.lower() in ['hello', 'hi', '你好']:
                bot_response = self.kernel.respond("GREETING") #TODO api接管？
            elif user_input.lower() in ['start planning', 'start', 'yes', 'yeah', '开始', '好的']:
                bot_response = self.kernel.respond("STARTPLANNING")  #进入planning模块
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
        self.general_query_kernel = init_aiml_kernel("career_query.aiml")
        self.mode = "general_query"  # Initial mode

    def process_aiml_formatting(self, text: str) -> str:
        text = text.replace('_br_', '\n\n')
        text = text.replace("_b_", "**")
        text = text.replace("_i_", "*")
        return text or "I'm not sure how to answer that. Try asking about careers, majors, or career preparation tips."

    def get_response(self, user_input: str) -> str:
        user_input_lower = user_input.strip().lower()

        # PART1 Planning mode
        if user_input_lower == "start planning":
            self.mode = "guided_planning"
            self.guided_chatbot.reset() # 启动guide
            return self.process_aiml_formatting(self.guided_chatbot.get_response(user_input)) 
       
        elif self.mode == "guided_planning" and user_input_lower == "start over": # 在guide中得到重启信号
            self.guided_chatbot.reset()
            return self.process_aiml_formatting(self.guided_chatbot.get_response(user_input)) # Restart guided
        
        elif self.mode == "guided_planning" and user_input_lower == "cancel planning" : #从guide转到general
            self.mode = "general_query"
            self.guided_chatbot.reset()
            return "Career planning cancelled. You are now in general query mode. Type 'start planning' to begin a new plan or 'hello' to ask general questions."
        
        # PART2 Ask general
        elif user_input_lower == "ask general":
            self.mode = "general_query"
            return self.process_aiml_formatting(self.general_query_kernel.respond("HELLO")) # Greet in general mode
        


        if self.mode == "guided_planning":
            return self.process_aiml_formatting(self.guided_chatbot.get_response(user_input))
        else: # general_query mode
            response = self.general_query_kernel.respond(user_input.strip().upper())
            return self.process_aiml_formatting(response)

    def reset_all(self):
        self.guided_chatbot.reset()
        self.mode = "general_query" # Reset to default mode
        print("INFO: All chatbot states reset.")

#尝试把careerchatbot和combinedchatbot合并到一起，不然逻辑太乱了
class Bot:
    def __init__(self, aiml_file="full_system.aiml"):
        self.kernel = aiml.Kernel()
        if os.path.exists(aiml_file):
            self.kernel.learn(aiml_file)

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
        
        #判断两种情形
        if user_input == "start planning":
            self.conversation_state = 1
            return self.process_aiml_formatting(self.kernel.respond("STARTPLANNING"))
        
        if user_input == "ask general":
            return self.process_aiml_formatting(self.kernel.respond("HELLO"))
        
        #处理中途退出
        if user_input == "start over":
            self.reset()
            return self.process_aiml_formatting(self.kernel.respond("STARTPLANNING"))        
        if user_input == "cancel planning":
            self.reset()
            return "Career planning cancelled. You can now ask general questions or start a new plan."
        
        # 生成plan的部分
        if self.conversation_state == 1:
            self.user_data['major'] = user_input
            self.conversation_state = 2
            template = self.kernel.respond("ASKINTERESTS")
            return template.format(major=self.user_data['major'])

        if self.conversation_state == 2:
            self.user_data['interests'] = user_input
            self.conversation_state = 3
            template = self.kernel.respond("ASKMBTI")
            return template.format(interests=self.user_data['interests'])

        if self.conversation_state == 3:
            if user_input in ["i don't know", "i dont know", "not sure", "不知道"]:
                self.user_data['mbti'] = "Unknown"
                self.conversation_state = 4
                return self.kernel.respond("ASKCHALLENGESSKIPMBTI")
            else:
                self.user_data['mbti'] = user_input.upper()
                self.conversation_state = 4
                template = self.kernel.respond("ASKCHALLENGES")
                return template.format(mbti=self.user_data['mbti'])

        if self.conversation_state == 4:
            self.user_data['challenges_input'] = user_input
            self.conversation_state = 5
            template = self.kernel.respond("CONFIRMINFO")
            return template.format(**self.user_data)

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



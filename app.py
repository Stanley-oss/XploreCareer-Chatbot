import sys
import os
import aiml
from expert_system import UserProfile, inference_engine, RULE_BASE
import io
from contextlib import redirect_stdout


CHALLENGE_MAP = {
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


def run_career_bot():
    kernel = aiml.Kernel()
    aiml_file = "career_dialogue.aiml"
    if os.path.exists(aiml_file):
        print(f"Learning {aiml_file}...")
        kernel.learn(aiml_file)
        print("Learning complete.")
    else:
        print(f"fault: {aiml_file} file can't be found")
        return

    conversation_state = 0
    user_data = {}

    print("\nCareer Planning Assistant is now running! Type 'hello' to start the conversation, or 'exit' to quit.")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        bot_response = ""
        command_to_kernel = ""


        if conversation_state == 0:
            if user_input.lower() in ['hello', 'hi']:
                command_to_kernel = "GREETING"
                bot_response = kernel.respond(command_to_kernel)
            elif user_input.lower() in ['start planning', 'start','yes','yeah']:
                command_to_kernel = "STARTPLANNING"
                bot_response = kernel.respond(command_to_kernel)
                if bot_response:
                    conversation_state = 1
            else:
                bot_response = "Sorry, I didn't understand. Please say 'hello' or 'start planning'."

        elif conversation_state == 1:
            user_data['major'] = user_input
            command_to_kernel = "ASKINTERESTS"
            template = kernel.respond(command_to_kernel)
            if template:
                bot_response = template.format(major=user_data['major'])
                conversation_state = 2


        elif conversation_state == 2:
            user_data['interests'] = user_input
            command_to_kernel = "ASKMBTI"
            template = kernel.respond(command_to_kernel)
            if template:
                bot_response = template.format(interests=user_data['interests'])
                conversation_state = 3

        elif conversation_state == 3:
            if user_input.lower() in ["i don't know", "i dont know", "not sure"]:
                user_data['mbti'] = "Unknown"
                command_to_kernel = "ASKCHALLENGESSKIPMBTI"
            else:
                user_data['mbti'] = user_input.upper()
                command_to_kernel = "ASKCHALLENGES"

            template = kernel.respond(command_to_kernel)
            if template and "mbti" in user_data:
                bot_response = template.format(mbti=user_data['mbti']) if user_data.get(
                    'mbti') != "Unknown" else template
            elif template:
                bot_response = template

            if template:
                conversation_state = 4

        elif conversation_state == 4:
            user_data['challenges_input'] = user_input
            command_to_kernel = "CONFIRMINFO"
            template = kernel.respond(command_to_kernel)
            if template:
                bot_response = template.format(**user_data)
                conversation_state = 5

        elif conversation_state == 5:
            if user_input.lower() == 'confirm':
                command_to_kernel = "FINALRESPONSE"
                print(kernel.respond(command_to_kernel))


                user_major = user_data.get('major', '')
                user_interests_str = user_data.get('interests', '')
                user_mbti = user_data.get('mbti', '')
                user_challenges_str = user_data.get('challenges_input', '')

                user_interests = [interest.strip() for interest in user_interests_str.split(',')]
                if user_mbti == 'Unknown': user_mbti = ""
                challenge_numbers = [num.strip() for num in user_challenges_str.split(',')]
                user_challenges = [CHALLENGE_MAP[num] for num in challenge_numbers if num in CHALLENGE_MAP]

                profile = UserProfile(major=user_major, interests=user_interests, mbti=user_mbti,
                                      challenges=user_challenges)
                f = io.StringIO()
                with redirect_stdout(f):
                    inference_engine(profile, RULE_BASE)
                    final_abilities = profile.abilities.sort_values(ascending=False)
                    print("\n--- Final Ability Weights Analysis ---")
                    for ability, score in final_abilities.items():
                        print(f"{ability:<25} {score:+.2f}")
                    print("=" * 35)
                analysis_result = f.getvalue()
                print(analysis_result)
                print("[System] Analysis complete. The program will now exit.")
                break

            elif user_input.lower() == 'start over':
                user_data = {}
                command_to_kernel = "STARTPLANNING"
                bot_response = kernel.respond(command_to_kernel)
                conversation_state = 1
            else:
                bot_response = "Sorry, I didn't understand. Please say 'confirm' or 'start over'."


        if not bot_response:
            print(f"--- DEBUG INFO ---")
            print(f"Current State: {conversation_state}")
            print(f"User Input: '{user_input}'")
            print(f"Command to Kernel: '{command_to_kernel}'")
            print(f"Kernel Raw Response: '{kernel.respond(command_to_kernel)}'")
            print(f"--------------------")
            bot_response = "Sorry, an internal error occurred."

        print("Bot:", bot_response)


if __name__ == "__main__":
    run_career_bot()
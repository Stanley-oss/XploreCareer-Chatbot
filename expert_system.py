import pandas as pd
import numpy as np
from typing import List, Dict, Any

# --- Step 1: Knowledge Base Ultimate Edition ---


ABILITY_COLUMNS = [
    'Mathematical Skills', 'Programming Ability', 'Creativity', 'Analytical Skills', 'Communication Skills',
    'Leadership Skills', 'Business Acumen', 'Problem-Solving', 'Teamwork', 'Adaptability'
]

# ============== Category A: Professional Rules ==============
RULE_BASE = [

    {'type': 'Major', 'conditions': ['Computer Science and Technology', 'CST'],
     'effects': {'Programming Ability': 0.5, 'Problem-Solving': 0.4, 'Analytical Skills': 0.4,
                 'Mathematical Skills': 0.3}},
    {'type': 'Major', 'conditions': ['Software Engineering', 'SWE'],
     'effects': {'Programming Ability': 0.5, 'Problem-Solving': 0.4, 'Teamwork': 0.3, 'Analytical Skills': 0.1}},
    {'type': 'Major', 'conditions': ['Artificial Intelligence', 'AIT'],
     'effects': {'Mathematical Skills': 0.5, 'Programming Ability': 0.4, 'Analytical Skills': 0.4, 'Creativity': 0.3}},
    {'type': 'Major', 'conditions': ['Data Science & Big Data Technology', 'DSC'],
     'effects': {'Mathematical Skills': 0.4, 'Analytical Skills': 0.4, 'Programming Ability': 0.3,
                 'Business Acumen': 0.2}},
    {'type': 'Major', 'conditions': ['Cyberspace Security', 'CYS'],
     'effects': {'Problem-Solving': 0.4, 'Analytical Skills': 0.3, 'Programming Ability': 0.3, 'Teamwork': -0.1}},
    {'type': 'Major', 'conditions': ['Digital Media Technology', 'DMT'],
     'effects': {'Creativity': 0.5, 'Teamwork': 0.3, 'Programming Ability': 0.2, 'Analytical Skills': -0.1}},
    {'type': 'Major', 'conditions': ['E-commerce', 'ECM'],
     'effects': {'Business Acumen': 0.5, 'Communication Skills': 0.3, 'Adaptability': 0.2, 'Teamwork': 0.2}},
    {'type': 'Major', 'conditions': ['Accounting', 'ACC'],
     'effects': {'Analytical Skills': 0.4, 'Business Acumen': 0.3, 'Mathematical Skills': 0.2, 'Creativity': -0.2}},
    {'type': 'Major', 'conditions': ['Finance', 'FIN'],
     'effects': {'Business Acumen': 0.5, 'Mathematical Skills': 0.4, 'Analytical Skills': 0.3,
                 'Communication Skills': 0.2}},
    {'type': 'Major', 'conditions': ['International Business', 'IBU'],
     'effects': {'Communication Skills': 0.5, 'Business Acumen': 0.4, 'Adaptability': 0.3, 'Leadership Skills': 0.1}},
    {'type': 'Major', 'conditions': ['Chinese Studies', 'CHS'],
     'effects': {'Communication Skills': 0.4, 'Creativity': 0.3, 'Analytical Skills': 0.1,
                 'Programming Ability': -0.4}},
    {'type': 'Major', 'conditions': ['English Studies', 'ENG'],
     'effects': {'Communication Skills': 0.5, 'Adaptability': 0.2, 'Teamwork': 0.2, 'Mathematical Skills': -0.2}},
    {'type': 'Major', 'conditions': ['Advertising', 'ADT'],
     'effects': {'Creativity': 0.5, 'Communication Skills': 0.4, 'Business Acumen': 0.3, 'Teamwork': 0.2}},
    {'type': 'Major', 'conditions': ['Journalism', 'JRN'],
     'effects': {'Communication Skills': 0.5, 'Adaptability': 0.3, 'Analytical Skills': 0.2, 'Problem-Solving': 0.2}},
    {'type': 'Major', 'conditions': ['Communication Studies', 'COS'],
     'effects': {'Communication Skills': 0.5, 'Creativity': 0.4, 'Teamwork': 0.3, 'Adaptability': 0.2}},
    {'type': 'Major', 'conditions': ['Electronic and Information Engineering', 'EEE'],
     'effects': {'Problem-Solving': 0.4, 'Mathematical Skills': 0.4, 'Analytical Skills': 0.3,
                 'Programming Ability': 0.2}},
    {'type': 'Major', 'conditions': ['Chemical Engineering and Technology', 'CME'],
     'effects': {'Analytical Skills': 0.5, 'Problem-Solving': 0.4, 'Mathematical Skills': 0.3, 'Teamwork': 0.1}},
    {'type': 'Major', 'conditions': ['New Energy Science and Engineering', 'EGE'],
     'effects': {'Problem-Solving': 0.4, 'Mathematical Skills': 0.3, 'Analytical Skills': 0.3, 'Adaptability': 0.2}},
    {'type': 'Major', 'conditions': ['Mathematics and Applied Mathematics', 'MAT'],
     'effects': {'Mathematical Skills': 0.6, 'Analytical Skills': 0.5, 'Problem-Solving': 0.4,
                 'Communication Skills': -0.2}},
    {'type': 'Major', 'conditions': ['Physics', 'PHY'],
     'effects': {'Mathematical Skills': 0.5, 'Analytical Skills': 0.5, 'Problem-Solving': 0.4,
                 'Communication Skills': -0.1}},
    {'type': 'Major', 'conditions': ['Marine Technology', 'MBT'],
     'effects': {'Problem-Solving': 0.4, 'Analytical Skills': 0.3, 'Adaptability': 0.3, 'Programming Ability': 0.2}},
    {'type': 'Major', 'conditions': ['Marine Science', 'MEC'],
     'effects': {'Analytical Skills': 0.5, 'Problem-Solving': 0.3, 'Adaptability': 0.2, 'Teamwork': 0.2}},
    {'type': 'Major', 'conditions': ['Traditional Chinese Medicine', 'TCM'],
     'effects': {'Analytical Skills': 0.4, 'Communication Skills': 0.3, 'Problem-Solving': 0.2, 'Adaptability': 0.2}},

    # ============== Category B: Rules for Special Interests and Talents ==============
    {'type': 'Interest', 'conditions': ['programming contest', 'coding', 'developing'],
     'effects': {'Programming Ability': 0.4, 'Problem-Solving': 0.3, 'Mathematical Skills': 0.2}},
    {'type': 'Interest', 'conditions': ['math modeling', 'modeling'],
     'effects': {'Mathematical Skills': 0.4, 'Analytical Skills': 0.3, 'Problem-Solving': 0.3}},
    {'type': 'Interest', 'conditions': ['scientific research', 'research', 'lab work'],
     'effects': {'Analytical Skills': 0.5, 'Problem-Solving': 0.3, 'Teamwork': -0.1}},
    {'type': 'Interest', 'conditions': ['strategy games', 'chess', 'logic puzzles'],
     'effects': {'Analytical Skills': 0.3, 'Problem-Solving': 0.3, 'Adaptability': -0.1}},
    {'type': 'Interest', 'conditions': ['electronics DIY', 'model making'],
     'effects': {'Problem-Solving': 0.3, 'Creativity': 0.2, 'Analytical Skills': 0.1}},
    {'type': 'Interest', 'conditions': ['reading books', 'visiting museums'],
     'effects': {'Analytical Skills': 0.2, 'Adaptability': 0.1}},
    {'type': 'Interest', 'conditions': ['attending academic lectures'],
     'effects': {'Adaptability': 0.2, 'Analytical Skills': 0.1}},
    {'type': 'Interest', 'conditions': ['debating', 'public speaking', 'hosting'],
     'effects': {'Communication Skills': 0.5, 'Analytical Skills': 0.3, 'Leadership Skills': 0.2}},
    {'type': 'Interest', 'conditions': ['student union', 'club management', 'organizing events'],
     'effects': {'Leadership Skills': 0.4, 'Teamwork': 0.3, 'Business Acumen': 0.2}},
    {'type': 'Interest', 'conditions': ['team sports', 'basketball', 'football'],
     'effects': {'Teamwork': 0.4, 'Adaptability': 0.3, 'Leadership Skills': 0.2}},
    {'type': 'Interest', 'conditions': ['volunteering', 'social work'],
     'effects': {'Teamwork': 0.4, 'Communication Skills': 0.3, 'Adaptability': 0.3}},
    {'type': 'Interest', 'conditions': ['content creation', 'podcast', 'vlog'],
     'effects': {'Communication Skills': 0.4, 'Creativity': 0.3, 'Business Acumen': 0.1}},
    {'type': 'Interest', 'conditions': ['online gaming', 'MMORPG', 'esports'],
     'effects': {'Teamwork': 0.3, 'Problem-Solving': 0.2, 'Adaptability': 0.2}},
    {'type': 'Interest', 'conditions': ['art', 'drawing', 'design'],
     'effects': {'Creativity': 0.5, 'Adaptability': 0.2, 'Analytical Skills': -0.1}},
    {'type': 'Interest', 'conditions': ['creative writing', 'writing novels'],
     'effects': {'Creativity': 0.4, 'Communication Skills': 0.3}},
    {'type': 'Interest', 'conditions': ['photography', 'filmmaking'],
     'effects': {'Creativity': 0.3, 'Adaptability': 0.1}},
    {'type': 'Interest', 'conditions': ['music', 'instrument', 'singing', 'dancing'],
     'effects': {'Creativity': 0.3, 'Teamwork': 0.1, 'Adaptability': 0.2}},
    {'type': 'Interest', 'conditions': ['cooking', 'baking'],
     'effects': {'Creativity': 0.2, 'Problem-Solving': 0.1, 'Adaptability': 0.2}},
    {'type': 'Interest', 'conditions': ['fashion', 'makeup'], 'effects': {'Creativity': 0.4, 'Business Acumen': 0.1}},
    {'type': 'Interest', 'conditions': ['board games', 'TTRPG'],
     'effects': {'Creativity': 0.3, 'Teamwork': 0.3, 'Communication Skills': 0.2, 'Problem-Solving': 0.2}},
    {'type': 'Interest', 'conditions': ['startup', 'entrepreneurship'],
     'effects': {'Business Acumen': 0.5, 'Leadership Skills': 0.3, 'Problem-Solving': 0.3}},
    {'type': 'Interest', 'conditions': ['investing', 'stock market'],
     'effects': {'Business Acumen': 0.5, 'Analytical Skills': 0.3, 'Mathematical Skills': 0.2}},
    {'type': 'Interest', 'conditions': ['traveling', 'exploring'],
     'effects': {'Adaptability': 0.4, 'Communication Skills': 0.1, 'Problem-Solving': 0.1}},
    {'type': 'Interest', 'conditions': ['learning languages'],
     'effects': {'Communication Skills': 0.3, 'Adaptability': 0.2}},

    # ============== Type C: Personality Rules ==============
    {'type': 'MBTI', 'condition': 'E',
     'effects': {'Communication Skills': 0.3, 'Teamwork': 0.2, 'Leadership Skills': 0.1}},
    {'type': 'MBTI', 'condition': 'I',
     'effects': {'Analytical Skills': 0.2, 'Problem-Solving': 0.1, 'Communication Skills': -0.2}},
    {'type': 'MBTI', 'condition': 'S', 'effects': {'Problem-Solving': 0.2, 'Teamwork': 0.1, 'Creativity': -0.2}},
    {'type': 'MBTI', 'condition': 'N', 'effects': {'Creativity': 0.3, 'Adaptability': 0.1, 'Analytical Skills': 0.1}},
    {'type': 'MBTI', 'condition': 'T',
     'effects': {'Analytical Skills': 0.3, 'Problem-Solving': 0.2, 'Business Acumen': 0.1}},
    {'type': 'MBTI', 'condition': 'F',
     'effects': {'Communication Skills': 0.3, 'Teamwork': 0.2, 'Analytical Skills': -0.1}},
    {'type': 'MBTI', 'condition': 'J',
     'effects': {'Leadership Skills': 0.2, 'Problem-Solving': 0.1, 'Adaptability': -0.2}},
    {'type': 'MBTI', 'condition': 'P', 'effects': {'Adaptability': 0.3, 'Creativity': 0.2, 'Leadership Skills': -0.1}},

    # ============== Category D: Challenging Situational Rules ==============
    {'type': 'Challenge', 'condition': 'dislikes group projects',
     'suppression_factors': {'Teamwork': 0.2, 'Communication Skills': 0.7, 'Leadership Skills': 0.8},
     'direct_effects': {'Teamwork': -0.2}},
    {'type': 'Challenge', 'condition': 'dislikes public speaking or presentations',
     'suppression_factors': {'Communication Skills': 0.1, 'Leadership Skills': 0.5},
     'direct_effects': {'Communication Skills': -0.3}},
    {'type': 'Challenge', 'condition': 'hard to come up with new, original ideas',
     'suppression_factors': {'Creativity': 0.1}, 'direct_effects': {'Creativity': -0.3, 'Problem-Solving': -0.1}},
    {'type': 'Challenge', 'condition': 'gets a headache from complex data or math',
     'suppression_factors': {'Mathematical Skills': 0.1, 'Analytical Skills': 0.4},
     'direct_effects': {'Mathematical Skills': -0.3}},
    {'type': 'Challenge', 'condition': 'prefers clear instructions over ambiguous tasks',
     'suppression_factors': {'Adaptability': 0.3, 'Creativity': 0.5},
     'direct_effects': {'Adaptability': -0.2, 'Problem-Solving': -0.1}},
    {'type': 'Challenge', 'condition': 'tends to lose the big picture when facing too much information',
     'suppression_factors': {'Analytical Skills': 0.2},
     'direct_effects': {'Analytical Skills': -0.2, 'Problem-Solving': -0.2}},
    {'type': 'Challenge', 'condition': 'not interested in business operations or how companies make profit',
     'suppression_factors': {'Business Acumen': 0.1}, 'direct_effects': {'Business Acumen': -0.4}},
    {'type': 'Challenge', 'condition': 'gets anxious under pressure or tight deadlines',
     'suppression_factors': {'Adaptability': 0.4, 'Problem-Solving': 0.6},
     'direct_effects': {'Adaptability': -0.3, 'Problem-Solving': -0.1}},
    {'type': 'Challenge', 'condition': 'finds it difficult to persuade others',
     'suppression_factors': {'Communication Skills': 0.4, 'Leadership Skills': 0.4},
     'direct_effects': {'Communication Skills': -0.2, 'Business Acumen': -0.1}},
    {'type': 'Challenge', 'condition': 'prefers to complete tasks independently rather than leading a team',
     'suppression_factors': {'Leadership Skills': 0.2, 'Teamwork': 0.5}, 'direct_effects': {'Leadership Skills': -0.3}},
    {'type': 'Challenge', 'condition': 'tends to procrastinate, deadlines are the main motivation',
     'suppression_factors': {'Problem-Solving': 0.7, 'Leadership Skills': 0.8},
     'direct_effects': {'Adaptability': -0.2}},
    {'type': 'Challenge', 'condition': 'afraid of or dislikes handling interpersonal conflicts',
     'suppression_factors': {'Communication Skills': 0.6, 'Leadership Skills': 0.5, 'Teamwork': 0.7},
     'direct_effects': {'Communication Skills': -0.1, 'Leadership Skills': -0.2}},
    {'type': 'Challenge', 'condition': 'gets bored easily by repetitive, routine tasks', 'suppression_factors': {},
     'direct_effects': {'Adaptability': -0.4, 'Analytical Skills': -0.1}},
    {'type': 'Challenge', 'condition': 'dislikes networking or actively building new connections',
     'suppression_factors': {'Communication Skills': 0.7, 'Business Acumen': 0.6},
     'direct_effects': {'Communication Skills': -0.2}},
    {'type': 'Challenge', 'condition': 'afraid of making mistakes, tends to be a perfectionist',
     'suppression_factors': {'Creativity': 0.5, 'Adaptability': 0.4},
     'direct_effects': {'Problem-Solving': 0.1, 'Analytical Skills': 0.1, 'Creativity': -0.1}},
    {'type': 'Challenge', 'condition': 'finds it hard to maintain focus for long periods',
     'suppression_factors': {'Analytical Skills': 0.6, 'Problem-Solving': 0.7},
     'direct_effects': {'Adaptability': -0.2}},
    {'type': 'Challenge', 'condition': 'not good at reporting work to superiors or clients',
     'suppression_factors': {'Communication Skills': 0.6, 'Business Acumen': 0.7},
     'direct_effects': {'Communication Skills': -0.2}},
    {'type': 'Challenge', 'condition': 'struggles with purely theoretical concepts, needs hands-on practice',
     'suppression_factors': {'Analytical Skills': 0.7},
     'direct_effects': {'Mathematical Skills': -0.2, 'Analytical Skills': -0.1}},
    {'type': 'Challenge', 'condition': 'hesitates when making decisions',
     'suppression_factors': {'Leadership Skills': 0.6, 'Problem-Solving': 0.6},
     'direct_effects': {'Adaptability': -0.2}},
]


# --- Step 2: Introduce the smart user profile class with ‘diminishing marginal returns’ ---
class UserProfile:
    def __init__(self, major: str, interests: List[str], mbti: str, challenges: List[str]):
        self.major = major
        self.interests = interests
        self.mbti = mbti.upper().strip() if mbti else ""
        self.challenges = challenges
        self.abilities = pd.Series(0.0, index=ABILITY_COLUMNS)

    def apply_effects(self, effects: Dict[str, float]):
        for ability, effect in effects.items():
            if ability in self.abilities.index:
                current_score = self.abilities[ability]

                remaining_space = 1 - abs(current_score)

                adjusted_effect = effect * remaining_space

                self.abilities[ability] += adjusted_effect

    def apply_suppression(self, factors: Dict[str, float]):
        for ability, factor in factors.items():
            if ability in self.abilities.index and self.abilities[ability] > 0:
                self.abilities[ability] *= factor

    def normalize_scores(self):
        self.abilities = np.clip(self.abilities, -1.0, 1.0)


# --- Step 3: Reasoning machine and main function (remains unchanged) ---
def inference_engine(user_profile: UserProfile, rules: List[Dict[str, Any]]):
    for rule in rules:
        if rule['type'] == 'Major' and any(
                keyword.lower() in user_profile.major.lower() for keyword in rule['conditions']):
            user_profile.apply_effects(rule['effects'])
            break

    for interest in user_profile.interests:
        for rule in rules:
            if rule['type'] == 'Interest' and any(
                    keyword.lower() in interest.lower() for keyword in rule['conditions']):
                user_profile.apply_effects(rule['effects'])

    if user_profile.mbti:
        for letter in user_profile.mbti:
            for rule in rules:
                if rule['type'] == 'MBTI' and letter == rule['condition']:
                    user_profile.apply_effects(rule['effects'])

    for challenge in user_profile.challenges:
        for rule in rules:
            if rule['type'] == 'Challenge' and challenge == rule['condition']:
                if 'suppression_factors' in rule:
                    user_profile.apply_suppression(rule['suppression_factors'])
                if 'direct_effects' in rule:
                    user_profile.apply_effects(rule['direct_effects'])

    user_profile.normalize_scores()


def main():
    print(f"=== User Ability Assessment System ===")
    print(f"Major: {user_major}")
    print(f"Interests: {', '.join(user_interests)}")
    print(f"MBTI: {user_mbti}")
    print(f"Selected Challenges: {', '.join(user_challenges)}")
    print("=" * 50)

    user_profile = UserProfile(
        major=user_major,
        interests=user_interests,
        mbti=user_mbti,
        challenges=user_challenges
    )

    inference_engine(user_profile, RULE_BASE)

    print("\n--- Final Ability Weights ---")
    final_abilities = user_profile.abilities.sort_values(ascending=False)
    for ability, score in final_abilities.items():
        print(f"{ability:<25} {score:+.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

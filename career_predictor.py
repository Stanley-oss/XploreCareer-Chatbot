import pandas as pd
import numpy as np
import re
import nltk
import math
import os
from typing import List
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from time import perf_counter

DEBUG = True


class CareerPredictor:
    def __init__(self):
        try:
            file = pd.read_csv("weights.csv")
            data = file.iloc[:, 1:].values
            self.professions = file.iloc[:, 0].values
            self.feature_matrix = np.array(data, dtype=float)
        except Exception as e:
            print(f"Error: unable to load weights.csv file. {str(e)}")
            self.professions = np.array(["Software Engineer", "Data Scientist", "Manager", "Designer", "Analyst"])
            self.feature_matrix = np.random.rand(5, 10)

        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)
            nltk.download("vader_lexicon", download_dir=nltk_data_dir)

        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vader = SentimentIntensityAnalyzer()

        self.aspects = [
            "Mathematical Skills", "Programming Ability", "Creativity", "Analytical Skills",
            "Communication Skills", "Leadership Skills", "Business Acumen", "Problem-Solving",
            "Teamwork", "Adaptability"
        ]

        self.aspect_embs = self.st_model.encode(self.aspects, convert_to_tensor=True)

        self.gamma = 1

        self.n_dim = len(self.aspects)
        self.sigma0 = 1.0
        self.n_samples = 1000
        self.burnin = 200
        self.step_size = 0.05
        self.min_sigma = 0.01
        self.max_sigma = 2.0

        print("Predictor initialized.")

    def log_posterior(self, W, data, sigma):
        """
        计算对数后验概率
        """
        try:
            log_prior = -0.5 * np.sum(W ** 2) / (self.sigma0 ** 2)

            log_likelihood = 0
            for j, w in data:
                if 0 <= j < len(W):
                    residual = w - W[j]
                    log_likelihood += -0.5 * (residual ** 2) / (sigma ** 2)

            result = log_prior + log_likelihood

            if np.isnan(result) or np.isinf(result):
                return -np.inf

            return result

        except Exception:
            return -np.inf

    def mcmc(self, data):
        if not data or all(abs(w) < 1e-10 for _, w in data):
            if DEBUG:
                print("All tendency scores are zero, using direct mapping")
            return np.array([w for _, w in data])

        W = np.zeros(self.n_dim)
        samples = []
        accepted_count = 0
        sigma = 1.0

        step_size = self.step_size
        target_acceptance = 0.44

        for iteration in range(self.n_samples):
            W_prop = W + np.random.normal(0, step_size, self.n_dim)

            try:
                log_alpha = (
                        self.log_posterior(W_prop, data, sigma)
                        - self.log_posterior(W, data, sigma)
                )
                log_alpha = np.clip(log_alpha, -50, 10)
                # Metropolis-Hastings
                if log_alpha > 0 or np.random.rand() < np.exp(log_alpha):
                    W = W_prop
                    accepted_count += 1

            except Exception:
                pass

            if iteration >= self.burnin:
                samples.append(W.copy())

            if iteration > 0 and iteration % 100 == 0:
                current_acceptance = accepted_count / iteration
                if current_acceptance < target_acceptance * 0.8:
                    step_size *= 0.9
                elif current_acceptance > target_acceptance * 1.2:
                    step_size *= 1.1
                step_size = np.clip(step_size, 0.001, 0.5)

        if samples:
            samples = np.array(samples)
            posterior_mean = samples.mean(axis=0)
            if len(data) > 0:
                residuals = np.array([w - posterior_mean[i] for i, w in data if 0 <= i < len(posterior_mean)])
                if len(residuals) > 0:
                    empirical_std = np.std(residuals)
                    shrinkage = max(0.1, 0.8 / np.sqrt(len(data)))
                    sigma = np.clip(max(empirical_std, shrinkage), self.min_sigma, self.max_sigma)

            if DEBUG:
                acceptance_rate = accepted_count / self.n_samples
                print(f"MCMC completed: sigma={sigma:.3f}, mean={posterior_mean.mean():.3f}, "
                      f"acceptance_rate={acceptance_rate:.3f}, final_step_size={step_size:.4f}")

            return posterior_mean
        else:
            if DEBUG:
                print("Warning: No valid MCMC samples, using direct tendency scores")
            return np.array([w for _, w in data])

    def predict(self, text: str) -> dict:
        """
        进行职业预测：输入用户的回答，返回预测的前10个职业及其概率
        """
        if not text.strip():
            return {profession: 0.0 for profession in self.professions[:min(10, len(self.professions))]}

        try:
            sents = [s.strip() for s in re.split(r"[.;!?]\s*", text) if s.strip()]
            if not sents:
                return {profession: 0.0 for profession in self.professions[:min(10, len(self.professions))]}

            sent_embs = self.st_model.encode(sents, convert_to_tensor=True)
            cos_sim = util.cos_sim(sent_embs, self.aspect_embs).cpu().numpy()
            senti_scores = [self.vader.polarity_scores(s)["compound"] for s in sents]
            print(f"Sentiment scores: {senti_scores}")

            tendency: List[tuple] = []
            for i in range(len(self.aspects)):
                w = max(0, 0, cos_sim[0][i])
                raw_score = w * senti_scores[0]
                score = max(-1.0, min(1.0, raw_score))
                score = math.copysign(math.pow(abs(score), self.gamma) * 2, raw_score)
                score = max(-2.0, min(2.0, score))
                tendency.append((i, score))

            posterior = self.mcmc(tendency)

            scores = self.feature_matrix.dot(posterior)

            # 归一化
            if len(scores) > 1:
                min_score = np.min(scores)
                max_score = np.max(scores)
                score_range = max_score - min_score

                if score_range > 1e-10:
                    scores = (scores - min_score) / score_range
                else:
                    scores = np.ones_like(scores) * 0.5
            else:
                scores = np.array([0.5])  # 单个职业时的默认值

            result = dict()
            num_professions = min(10, len(self.professions))
            top_indices = np.argsort(-scores)[:num_professions]

            for idx in top_indices:
                if idx < len(self.professions):
                    result[self.professions[idx]] = float(scores[idx])

            if DEBUG:
                print(f"Predicted professions: {result}")

            return result

        except Exception as e:
            if DEBUG:
                print(f"Error in prediction: {str(e)}")
            # 返回默认结果
            return {profession: 0.1 for profession in self.professions[:min(10, len(self.professions))]}

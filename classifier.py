# classifier.py
# Naïve Bayes sentiment classifier implemented from scratch.
#
# THEORY OVERVIEW:
# Naïve Bayes classifies text using Bayes' theorem:
#   P(class | words) ∝ P(class) * ∏ P(word | class)
#
# We make the "naïve" assumption that each word is independent of the others,
# which lets us multiply individual word probabilities together.
#
# To avoid floating point underflow from multiplying many small probabilities,
# we work in log space and sum instead:
#   log P(class | words) ∝ log P(class) + Σ log P(word | class)
#
# LAPLACE SMOOTHING:
# If a word appears in the test text but never in training, P(word|class) = 0
# and the whole product collapses to 0. Laplace smoothing adds a small count
# (alpha=1) to every word so no probability is ever exactly zero.

import re
import math
from collections import defaultdict
from data.training_data import TRAINING_DATA


def tokenize(text):
    """
    Convert a raw string into a list of lowercase word tokens.
    We strip punctuation and split on whitespace.
    Example: "I loved it!" -> ["i", "loved", "it"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-alphabetic characters
    return text.split()


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        """
        alpha: Laplace smoothing parameter (default 1.0 = add-one smoothing).
               Higher alpha = more smoothing = less influence from rare words.
        """
        self.alpha = alpha

        # word_counts[class][word] = number of times word appears in that class
        self.word_counts = defaultdict(lambda: defaultdict(int))

        # class_totals[class] = total number of word tokens seen in that class
        self.class_totals = defaultdict(int)

        # class_doc_counts[class] = number of training documents in that class
        self.class_doc_counts = defaultdict(int)

        # vocabulary = set of all unique words seen during training
        self.vocabulary = set()

        self.classes = []  # list of unique class labels
        self.total_docs = 0  # total number of training documents

    def train(self, data):
        """
        Learn word probabilities from labeled training data.
        data: list of (text, label) tuples
        """
        for text, label in data:
            tokens = tokenize(text)
            self.class_doc_counts[label] += 1
            for token in tokens:
                self.word_counts[label][token] += 1
                self.class_totals[label] += 1
                self.vocabulary.add(token)

        self.classes = list(self.class_doc_counts.keys())
        self.total_docs = sum(self.class_doc_counts.values())

    def _log_prob_word_given_class(self, word, cls):
        """
        Compute log P(word | class) with Laplace smoothing.

        Formula:
            P(word | class) = (count(word, class) + alpha)
                              / (total_words_in_class + alpha * vocab_size)

        Taking the log turns multiplication into addition later.
        """
        vocab_size = len(self.vocabulary)
        numerator = self.word_counts[cls][word] + self.alpha
        denominator = self.class_totals[cls] + self.alpha * vocab_size
        return math.log(numerator / denominator)

    def _log_prior(self, cls):
        """
        Compute log P(class) — how common this class is in training data.
        A balanced dataset gives equal priors (~0.5 each).
        """
        return math.log(self.class_doc_counts[cls] / self.total_docs)

    def predict(self, text):
        """
        Classify a text string. Returns:
          - label: the predicted class ("positive" or "negative")
          - confidence: probability of the predicted class (0-1)
          - explanation: list of dicts with per-word evidence for the UI
        """
        tokens = tokenize(text)

        # Accumulate log probabilities for each class
        log_scores = {}
        for cls in self.classes:
            log_scores[cls] = self._log_prior(cls)
            for token in tokens:
                log_scores[cls] += self._log_prob_word_given_class(token, cls)

        # Convert log scores to probabilities via softmax-style normalization.
        # We subtract the max log score first for numerical stability.
        max_score = max(log_scores.values())
        exp_scores = {cls: math.exp(score - max_score) for cls, score in log_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {cls: exp_scores[cls] / total for cls in self.classes}

        # Pick the class with the highest probability
        predicted_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_label]

        # Build per-word explanation for the frontend.
        # For each token, we compute how much it favors positive vs negative
        # by taking the difference in log probabilities.
        # Positive diff -> word leans positive; negative diff -> word leans negative.
        explanation = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            scores = {
                cls: self._log_prob_word_given_class(token, cls)
                for cls in self.classes
            }
            # diff > 0 means the word favors "positive", < 0 favors "negative"
            diff = scores.get("positive", 0) - scores.get("negative", 0)
            explanation.append({
                "word": token,
                "diff": round(diff, 4),
                # Raw log-prob for each class, for display
                "scores": {cls: round(scores[cls], 4) for cls in self.classes}
            })

        # Sort by absolute influence — most impactful words first
        explanation.sort(key=lambda x: abs(x["diff"]), reverse=True)

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {cls: round(p, 4) for cls, p in probabilities.items()},
            "explanation": explanation[:10],  # top 10 most influential words
        }


# Train a single shared instance on module load so Flask can import it directly
classifier = NaiveBayesClassifier(alpha=1.0)
classifier.train(TRAINING_DATA)
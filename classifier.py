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
#
# STOPWORD FILTERING:
# Common function words like "the", "a", "was" appear frequently in both
# classes and carry no sentiment signal. Removing them before training and
# prediction sharpens the classifier and cleans up the evidence table.

import re
import math
from collections import defaultdict
from data.training_data import TRAINING_DATA

# ── Stopword list ────────────────────────────────────────────────────────────
# A curated set of common English function words that carry no sentiment signal.
# These are filtered out before training AND before prediction so the
# vocabulary and evidence table only contain meaningful content words.

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "i", "my", "me", "we",
    "our", "you", "your", "he", "she", "his", "her", "they", "their",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "about", "after", "before", "between", "into",
    "through", "during", "up", "down", "out", "off", "over", "then",
    "there", "here", "also", "as", "at", "any", "am"
}


def tokenize(text, remove_stopwords=True):
    """
    Convert a raw string into a list of lowercase word tokens.
    We strip punctuation and split on whitespace.

    Args:
        text: raw input string
        remove_stopwords: if True, filter out words in STOPWORDS

    Returns:
        (tokens, filtered) where:
          tokens   — list of content words to use for classification
          filtered — list of words that were removed as stopwords
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()

    if not remove_stopwords:
        return words, []

    tokens = []
    filtered = []
    for word in words:
        if word in STOPWORDS:
            filtered.append(word)
        else:
            tokens.append(word)

    return tokens, filtered


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

        self.classes = []
        self.total_docs = 0

    def train(self, data):
        """
        Learn word probabilities from labeled training data.
        Stopwords are removed before training so they never enter the vocabulary.
        data: list of (text, label) tuples
        """
        for text, label in data:
            # Stopwords filtered out at training time too —
            # they won't be in the vocabulary at all
            tokens, _ = tokenize(text, remove_stopwords=True)
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
        """
        vocab_size = len(self.vocabulary)
        numerator = self.word_counts[cls][word] + self.alpha
        denominator = self.class_totals[cls] + self.alpha * vocab_size
        return math.log(numerator / denominator)

    def _log_prior(self, cls):
        """
        Compute log P(class) — how common this class is in training data.
        """
        return math.log(self.class_doc_counts[cls] / self.total_docs)

    def predict(self, text):
        """
        Classify a text string. Returns:
          - label: predicted class
          - confidence: probability of predicted class (0-1)
          - probabilities: dict of class -> probability
          - explanation: top 10 most influential content words
          - filtered: list of stopwords that were removed
        """
        tokens, filtered = tokenize(text, remove_stopwords=True)

        # Accumulate log probabilities for each class
        log_scores = {}
        for cls in self.classes:
            log_scores[cls] = self._log_prior(cls)
            for token in tokens:
                log_scores[cls] += self._log_prob_word_given_class(token, cls)

        # Convert log scores to probabilities via softmax-style normalization
        max_score = max(log_scores.values())
        exp_scores = {cls: math.exp(score - max_score) for cls, score in log_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {cls: exp_scores[cls] / total for cls in self.classes}

        predicted_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_label]

        # Build per-word explanation (content words only)
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
            diff = scores.get("positive", 0) - scores.get("negative", 0)
            explanation.append({
                "word": token,
                "diff": round(diff, 4),
                "scores": {cls: round(scores[cls], 4) for cls in self.classes}
            })

        explanation.sort(key=lambda x: abs(x["diff"]), reverse=True)

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {cls: round(p, 4) for cls, p in probabilities.items()},
            "explanation": explanation[:10],
            "filtered": filtered,  # stopwords removed — sent to frontend for display
        }


# Train a single shared instance on module load
classifier = NaiveBayesClassifier(alpha=1.0)
classifier.train(TRAINING_DATA)
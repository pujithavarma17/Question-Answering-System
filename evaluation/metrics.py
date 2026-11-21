"""Exact Match (accuracy) and F1 score for extractive question answering.

These follow the official SQuAD v2.0 evaluation script:
  * Answers are normalized (lower-case, remove punctuation, articles
    "a/an/the" and extra spaces) before comparing.
  * Exact Match (EM) = 1 if the normalized prediction equals any of the
    accepted answers, else 0.  The average EM is the model's accuracy.
  * F1 measures word overlap between prediction and answer:
        precision = common words / words in prediction
        recall    = common words / words in true answer
        F1        = 2 * precision * recall / (precision + recall)
    The best F1 over all accepted answers is used.
  * Unanswerable questions have no accepted answers. The model must
    predict "" (no answer) to get EM = F1 = 1 for them.
"""

import collections
import re
import string


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))


def f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()

    # If either is "no answer", F1 is 1 only when both are "no answer"
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def score_example(prediction, answers):
    """Score one prediction against a list of accepted answers.

    An empty `answers` list means the question is unanswerable.
    """
    truths = answers if answers else [""]
    return (
        max(exact_match(prediction, t) for t in truths),
        max(f1(prediction, t) for t in truths),
    )


def compute_metrics(predictions, references):
    """predictions: list of str; references: list of lists of str.

    Returns percentages for all questions, answerable ("has_answer")
    questions and unanswerable ("no_answer") questions.
    """
    groups = {"all": [], "has_answer": [], "no_answer": []}

    for prediction, answers in zip(predictions, references):
        em, f = score_example(prediction, answers)
        groups["all"].append((em, f))
        groups["has_answer" if answers else "no_answer"].append((em, f))

    results = {}
    for name, scores in groups.items():
        if not scores:
            continue
        prefix = "" if name == "all" else name + "_"
        results[prefix + "exact_match"] = round(100 * sum(s[0] for s in scores) / len(scores), 2)
        results[prefix + "f1"] = round(100 * sum(s[1] for s in scores) / len(scores), 2)
        results[prefix + "count"] = len(scores)

    return results

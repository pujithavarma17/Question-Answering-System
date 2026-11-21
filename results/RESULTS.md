# Evaluation results

EM = Exact Match (accuracy, %). F1 = word-overlap F1-score (%).

| Model | SQuAD v2 (500) EM | SQuAD v2 (500) F1 | Custom (60) EM | Custom (60) F1 | Document (20) EM | Document (20) F1 | No-answer F1 (SQuAD) | Sec/question (SQuAD) |
|---|---|---|---|---|---|---|---|---|
| `deepset/roberta-base-squad2` | 80.0 | 83.2 | 95.0 | 97.7 | 85.0 | 90.3 | 81.5 | 0.94 |
| `distilbert/distilbert-base-cased-distilled-squad` | 37.2 | 40.7 | 70.0 | 73.6 | 45.0 | 49.7 | 0.0 | 0.35 |
| `models/qa-distilbert-squad2` | 66.0 | 69.1 | 75.0 | 77.2 | 65.0 | 71.7 | 67.2 | 0.27 |

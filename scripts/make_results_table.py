"""Collect all evaluation results (results/*.json) into one comparison table.

Usage:  python scripts/make_results_table.py
Writes results/RESULTS.md and prints the table.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATASETS = [("squad", "SQuAD v2 (500)"), ("custom", "Custom (60)"), ("document", "Document (20)")]


def main():
    rows = {}
    for path in sorted(RESULTS.glob("*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        rows.setdefault(summary["model"], {})[summary["dataset"]] = summary

    header = "| Model | " + " | ".join(f"{name} EM | {name} F1" for _, name in DATASETS) + \
             " | No-answer F1 (SQuAD) | Sec/question (SQuAD) |"
    lines = [header, "|" + "---|" * (2 + 2 * len(DATASETS) + 1)]

    for model, results in rows.items():
        cells = [f"`{model}`"]
        for key, _ in DATASETS:
            r = results.get(key)
            cells += [f"{r['exact_match']:.1f}", f"{r['f1']:.1f}"] if r else ["–", "–"]
        squad = results.get("squad", {})
        cells.append(f"{squad['no_answer_f1']:.1f}" if "no_answer_f1" in squad else "–")
        cells.append(f"{squad['seconds_per_question']:.2f}" if squad else "–")
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    text = ("# Evaluation results\n\n"
            "EM = Exact Match (accuracy, %). F1 = word-overlap F1-score (%).\n\n"
            + table + "\n")
    (RESULTS / "RESULTS.md").write_text(text, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

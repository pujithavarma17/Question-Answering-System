"""Evaluate a question answering model with Exact Match (accuracy) and F1.

Datasets:
  squad     - random sample of the official SQuAD v2.0 validation set
  custom    - our hand-written dataset (data/custom_qa.json)
  document  - questions about a full document, answered with the complete
              app pipeline (chunking + retrieval + reader)

Examples (run from the project folder):
  python scripts/evaluate_qa.py --data custom
  python scripts/evaluate_qa.py --data squad --n 500
  python scripts/evaluate_qa.py --data document --model models/qa-distilbert-squad2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Hide model-loading progress bars and warnings so the results are readable
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from tqdm import tqdm  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.disable_progress_bar()

from evaluation.metrics import compute_metrics, score_example  # noqa: E402
from evaluation.qa_datasets import load_custom, load_document_qa, load_squad_v2  # noqa: E402
from qa_engine import DEFAULT_MODEL, answer_from_document, answer_from_passage  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Hugging Face model name or local folder")
    parser.add_argument("--data", choices=["squad", "custom", "document"], default="custom")
    parser.add_argument("--n", type=int, default=500,
                        help="number of SQuAD examples (only for --data squad)")
    args = parser.parse_args()

    if args.data == "squad":
        examples = load_squad_v2(args.n)
    elif args.data == "custom":
        examples = load_custom()
    else:
        examples = load_document_qa()

    print(f"Evaluating {args.model} on {len(examples)} '{args.data}' questions...")

    predictions = []
    details = []
    start = time.time()

    for ex in tqdm(examples):
        if args.data == "document":
            prediction = answer_from_document(ex["question"], ex["context"],
                                              model_name=args.model)["answer"]
        else:
            prediction, _ = answer_from_passage(ex["question"], ex["context"],
                                                model_name=args.model)

        em, f1 = score_example(prediction, ex["answers"])
        predictions.append(prediction)
        details.append({
            "question": ex["question"],
            "prediction": prediction,
            "answers": ex["answers"],
            "exact_match": em,
            "f1": round(f1, 3),
        })

    seconds = time.time() - start
    metrics = compute_metrics(predictions, [ex["answers"] for ex in examples])

    summary = {
        "model": args.model,
        "dataset": args.data,
        **metrics,
        "seconds_per_question": round(seconds / len(examples), 3),
    }

    print(json.dumps(summary, indent=2))

    wrong = [d for d in details if not d["exact_match"]]
    if wrong:
        print(f"\nQuestions not answered exactly ({len(wrong)}), first 10:")
        for d in wrong[:10]:
            print(f"  Q: {d['question']}\n     predicted: {d['prediction']!r}"
                  f"  expected: {d['answers'] or 'NO ANSWER'}  (F1 {d['f1']})")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    name = Path(args.model).name if Path(args.model).exists() else args.model.replace("/", "__")
    out_file = out_dir / f"{name}__{args.data}.json"
    out_file.write_text(json.dumps({"summary": summary, "details": details}, indent=2),
                        encoding="utf-8")
    print(f"\nSaved to {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Load the evaluation datasets in one common format:

    {"id": str, "question": str, "context": str, "answers": [str, ...]}

An empty answers list means the question is unanswerable.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_squad_v2(n=500, seed=42):
    """A random sample of the official SQuAD v2.0 validation set."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    return [
        {
            "id": row["id"],
            "question": row["question"],
            "context": row["context"],
            # SQuAD lists the same answer several times (one per annotator)
            "answers": sorted(set(row["answers"]["text"])),
        }
        for row in ds
    ]


def load_custom(path=DATA_DIR / "custom_qa.json"):
    """Our hand-written passage-level dataset (data/custom_qa.json)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = []

    for p_index, item in enumerate(data["data"]):
        context = item["context"]
        for q_index, qa in enumerate(item["qas"]):
            for answer in qa["answers"]:
                if answer not in context:
                    raise ValueError(
                        f"Answer {answer!r} is not in its context "
                        f"(passage {p_index}, question {q_index})"
                    )
            examples.append({
                "id": f"custom-{p_index}-{q_index}",
                "question": qa["question"],
                "context": context,
                "answers": qa["answers"],
                "domain": item.get("domain", ""),
            })

    return examples


def load_document_qa(path=DATA_DIR / "document_qa.json"):
    """Questions about one whole document (tests the full app pipeline)."""
    from preprocessing.text_cleaning import clean_text

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Clean the text exactly like the web app does
    document = clean_text((DATA_DIR / data["document"]).read_text(encoding="utf-8"))

    return [
        {
            "id": f"doc-{i}",
            "question": q["question"],
            "context": document,
            "answers": q["answers"],
        }
        for i, q in enumerate(data["questions"])
    ]

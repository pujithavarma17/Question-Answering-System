"""Fine-tune a pre-trained BERT-style model for question answering on SQuAD v2.0.

The model learns to predict the START and END position of the answer
inside the passage. For unanswerable questions it learns to point at the
special [CLS] token (position 0), which means "no answer".

Recommended: run on Google Colab with a free GPU
(see notebooks/Train_QA_Colab.ipynb). On a laptop CPU use a small
--train_samples value just to test that everything works.

Examples:
  # quick CPU test (~minutes)
  python scripts/train_qa.py --train_samples 500 --eval_samples 100 --epochs 1

  # full training on a GPU (~1 hour on a Colab T4)
  python scripts/train_qa.py --train_samples 0 --epochs 2
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator, set_seed)

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="distilbert-base-uncased",
                   help="pre-trained model to start from, e.g. bert-base-uncased")
    p.add_argument("--output_dir", default=str(ROOT / "models" / "qa-distilbert-squad2"))
    p.add_argument("--train_samples", type=int, default=0,
                   help="number of training questions to use (0 = all ~130k)")
    p.add_argument("--eval_samples", type=int, default=2000,
                   help="validation questions used to track loss during training")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push_to_hub", default="",
                   help="optional Hugging Face repo id, e.g. your-username/qa-distilbert-squad2")
    return p.parse_args()


def make_preprocess(tokenizer, max_length, stride):
    """Convert (question, context, answer) into token ids + start/end labels."""

    pad_right = tokenizer.padding_side == "right"

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]

        # Long contexts are split into several overlapping windows
        encodings = tokenizer(
            questions if pad_right else examples["context"],
            examples["context"] if pad_right else questions,
            truncation="only_second" if pad_right else "only_first",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = encodings.pop("overflow_to_sample_mapping")
        offsets = encodings.pop("offset_mapping")
        context_id = 1 if pad_right else 0
        start_positions, end_positions = [], []

        for i, offset in enumerate(offsets):
            answer = examples["answers"][sample_map[i]]
            sequence_ids = encodings.sequence_ids(i)

            # Unanswerable question -> label both positions as [CLS] (index 0)
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            context_start = sequence_ids.index(context_id)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(context_id)

            # Answer is not inside this window -> also [CLS]
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # Find the first and last token of the answer
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        encodings["start_positions"] = start_positions
        encodings["end_positions"] = end_positions
        return encodings

    return preprocess


def main():
    args = parse_args()
    set_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")

    # 1. Data
    raw = load_dataset("rajpurkar/squad_v2")
    train_raw = raw["train"].shuffle(seed=args.seed)
    if args.train_samples:
        train_raw = train_raw.select(range(min(args.train_samples, len(train_raw))))
    eval_raw = raw["validation"].shuffle(seed=args.seed + 1)
    eval_raw = eval_raw.select(range(min(args.eval_samples, len(eval_raw))))
    print(f"Training questions: {len(train_raw)}   validation questions: {len(eval_raw)}")

    # 2. Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)

    # 3. Preprocessing (tokenize + compute answer start/end token labels)
    preprocess = make_preprocess(tokenizer, args.max_length, args.stride)
    train_ds = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(preprocess, batched=True, remove_columns=eval_raw.column_names)

    # 4. Training
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    training_args = TrainingArguments(
        output_dir=str(ROOT / "checkpoints"),
        eval_strategy="epoch",       # report validation loss after every epoch
        save_strategy="no",          # we save the final model ourselves below
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=int(0.1 * total_steps),  # warm up for the first 10% of steps
        fp16=use_gpu,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60
    eval_metrics = trainer.evaluate()

    # 5. Save
    output_dir = Path(args.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    info = {
        "base_model": args.base_model,
        "train_questions": len(train_raw),
        "train_features": len(train_ds),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": torch.cuda.get_device_name(0) if use_gpu else "CPU",
        "training_minutes": round(minutes, 1),
        "final_eval_loss": round(eval_metrics["eval_loss"], 4),
    }
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Model saved to {output_dir}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Uploaded to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()

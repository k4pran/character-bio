import argparse
import json
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def load_label_maps(data_dir: Path):
    with (data_dir / "label2id.json").open("r", encoding="utf-8") as f:
        label2id = json.load(f)

    # Ensure all IDs are ints.
    label2id = {label: int(idx) for label, idx in label2id.items()}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label


def tokenize_and_align_labels(examples, tokenizer, max_length: int):
    """
    Converts word-level labels into tokenizer-level labels.

    We label only the first sub-token of each word.
    Extra sub-tokens get -100 so the loss ignores them.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned_labels = []

    for batch_index, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                # Ignore continuation subword tokens.
                label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def bio_spans(labels: List[str]) -> Set[Tuple[int, int, str]]:
    """
    Converts BIO labels into entity spans.

    Returns spans as:
      (start_index_inclusive, end_index_exclusive, entity_type)

    Example:
      B-CHARACTER I-CHARACTER O
      -> {(0, 2, "CHARACTER")}
    """
    spans = set()
    start = None
    entity_type = None

    for i, label in enumerate(labels + ["O"]):
        if label == "O" or label is None:
            if start is not None:
                spans.add((start, i, entity_type))
                start = None
                entity_type = None
            continue

        if "-" not in label:
            if start is not None:
                spans.add((start, i, entity_type))
                start = None
                entity_type = None
            continue

        prefix, current_type = label.split("-", 1)

        if prefix == "B":
            if start is not None:
                spans.add((start, i, entity_type))
            start = i
            entity_type = current_type

        elif prefix == "I":
            # Continue only if we are already inside the same entity type.
            # Otherwise treat stray I-* as a new B-*.
            if start is None or entity_type != current_type:
                if start is not None:
                    spans.add((start, i, entity_type))
                start = i
                entity_type = current_type

        else:
            if start is not None:
                spans.add((start, i, entity_type))
                start = None
                entity_type = None

    return spans


def make_compute_metrics(id2label):
    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        total_correct = 0
        total_predicted = 0
        total_gold = 0

        for pred_row, label_row in zip(predictions, labels):
            pred_labels = []
            gold_labels = []

            for pred_id, gold_id in zip(pred_row, label_row):
                if gold_id == -100:
                    continue

                pred_labels.append(id2label[int(pred_id)])
                gold_labels.append(id2label[int(gold_id)])

            pred_spans = bio_spans(pred_labels)
            gold_spans = bio_spans(gold_labels)

            total_correct += len(pred_spans & gold_spans)
            total_predicted += len(pred_spans)
            total_gold += len(gold_spans)

        precision = total_correct / total_predicted if total_predicted else 0.0
        recall = total_correct / total_gold if total_gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="models/character-ner")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label2id, id2label = load_label_maps(data_dir)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "validation.jsonl"),
            "test": str(data_dir / "test.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
    )

    trainer.train()

    print("\nValidation metrics:")
    print(trainer.evaluate(tokenized_dataset["validation"]))

    print("\nTest metrics:")
    print(trainer.evaluate(tokenized_dataset["test"]))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "label2id.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    with (output_dir / "id2label.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    print(f"\nSaved model to: {output_dir}")


if __name__ == "__main__":
    main()
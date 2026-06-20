import argparse
import json
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("device: CPU")


def load_label_maps(data_dir: Path):
    with (data_dir / "label2id.json").open("r", encoding="utf-8") as f:
        label2id = json.load(f)

    # Ensure all IDs are ints.
    label2id = {label: int(idx) for label, idx in label2id.items()}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label

def labels_to_ids(labels: List[str], label2id: dict, source: str = "") -> List[int]:
    ids = []

    for label in labels:
        if label not in label2id:
            raise ValueError(
                f"Unknown label '{label}' in {source}. "
                f"Known labels: {sorted(label2id.keys())}"
            )

        ids.append(label2id[label])

    return ids


def load_jsonl_file(path: Path, label2id: dict) -> List[dict]:
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if "tokens" not in row:
                raise ValueError(f"{path}:{line_number} missing 'tokens'")

            if "ner_tags" not in row:
                if "labels" not in row:
                    raise ValueError(
                        f"{path}:{line_number} missing both 'labels' and 'ner_tags'"
                    )

                row["ner_tags"] = labels_to_ids(
                    row["labels"],
                    label2id,
                    source=f"{path}:{line_number}"
                )

            if "labels" not in row:
                id2label = {idx: label for label, idx in label2id.items()}
                row["labels"] = [id2label[int(tag)] for tag in row["ner_tags"]]

            if len(row["tokens"]) != len(row["ner_tags"]):
                raise ValueError(
                    f"{path}:{line_number} token/tag length mismatch: "
                    f"{len(row['tokens'])} tokens vs {len(row['ner_tags'])} tags"
                )

            row.setdefault("meta_json", "{}")
            rows.append(row)

    return rows


def load_conll_file(path: Path, label2id: dict) -> List[dict]:
    """
    Reads a CoNLL-style file:

        # id = example-001
        token<TAB>label
        token<TAB>label

        # id = example-002
        token<TAB>label

    Blank line separates examples.
    Comment lines beginning with # are stored as metadata.
    """
    rows = []

    tokens: List[str] = []
    labels: List[str] = []
    metadata = {}

    def flush_example():
        nonlocal tokens, labels, metadata

        if not tokens:
            metadata = {}
            return

        if len(tokens) != len(labels):
            raise ValueError(
                f"{path} token/label length mismatch: "
                f"{len(tokens)} tokens vs {len(labels)} labels"
            )

        row = {
            "tokens": tokens,
            "labels": labels,
            "ner_tags": labels_to_ids(labels, label2id, source=str(path)),
            "meta_json": json.dumps(
                {
                    **metadata,
                    "source_file": str(path),
                },
                ensure_ascii=False,
            ),
        }

        rows.append(row)

        tokens = []
        labels = []
        metadata = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                flush_example()
                continue

            if line.startswith("#"):
                comment = line[1:].strip()

                if "=" in comment:
                    key, value = comment.split("=", 1)
                    metadata[key.strip()] = value.strip()
                else:
                    metadata.setdefault("comments", [])
                    metadata["comments"].append(comment)

                continue

            parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{line_number} expected at least token and label, got: {line}"
                )

            token = parts[0]
            label = parts[-1]

            if label not in label2id:
                raise ValueError(
                    f"{path}:{line_number} unknown label '{label}'. "
                    f"Known labels: {sorted(label2id.keys())}"
                )

            tokens.append(token)
            labels.append(label)

    flush_example()

    return rows


def find_conll_files(data_dir: Path, filename: str) -> List[Path]:
    return sorted(
        path for path in data_dir.rglob(filename)
        if path.is_file()
    )


def load_split_rows(data_dir: Path, split_name: str, label2id: dict) -> List[dict]:
    """
    Load a split in this order:

    1. For train: recursively load every **/train.conll
    2. For validation/test: recursively load every **/validation.conll or **/test.conll
    3. If no conll files found, fallback to data_dir/<split>.jsonl
    """
    conll_files = find_conll_files(data_dir, f"{split_name}.conll")

    if conll_files:
        print(f"\nFound {len(conll_files)} {split_name}.conll file(s):")
        for file in conll_files:
            print(f"  - {file}")

        rows = []
        for file in conll_files:
            rows.extend(load_conll_file(file, label2id))

        return rows

    jsonl_path = data_dir / f"{split_name}.jsonl"

    if jsonl_path.exists():
        print(f"\nUsing {jsonl_path}")
        return load_jsonl_file(jsonl_path, label2id)

    return []


def build_dataset_dict(data_dir: Path, label2id: dict, seed: int = 42) -> DatasetDict:
    train_rows = load_split_rows(data_dir, "train", label2id)
    validation_rows = load_split_rows(data_dir, "validation", label2id)
    test_rows = load_split_rows(data_dir, "test", label2id)

    if not train_rows:
        raise ValueError(
            f"No training data found. Expected at least one train.conll under {data_dir} "
            f"or {data_dir / 'train.jsonl'}"
        )

    print(f"\nLoaded train rows: {len(train_rows)}")
    print(f"Loaded validation rows: {len(validation_rows)}")
    print(f"Loaded test rows: {len(test_rows)}")

    # If validation and test exist, use them directly.
    if validation_rows and test_rows:
        return DatasetDict(
            {
                "train": Dataset.from_list(train_rows),
                "validation": Dataset.from_list(validation_rows),
                "test": Dataset.from_list(test_rows),
            }
        )

    # Otherwise, auto-split the merged train data.
    print(
        "\nNo complete validation/test split found. "
        "Auto-splitting merged train data into 80% train, 10% validation, 10% test."
    )

    full = Dataset.from_list(train_rows)

    train_temp = full.train_test_split(
        test_size=0.2,
        seed=seed,
        shuffle=True,
    )

    validation_test = train_temp["test"].train_test_split(
        test_size=0.5,
        seed=seed,
        shuffle=True,
    )

    return DatasetDict(
        {
            "train": train_temp["train"],
            "validation": validation_test["train"],
            "test": validation_test["test"],
        }
    )

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

    dataset = build_dataset_dict(
        data_dir=data_dir,
        label2id=label2id,
        seed=42,
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
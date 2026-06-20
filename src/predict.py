import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_id2label(model_dir: Path):
    with (model_dir / "id2label.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return {int(k): v for k, v in raw.items()}


def merge_character_spans(words, labels):
    characters = []
    current_tokens = []

    for word, label in zip(words, labels):
        if label == "B-CHARACTER":
            if current_tokens:
                characters.append(" ".join(current_tokens))
            current_tokens = [word]

        elif label == "I-CHARACTER":
            if current_tokens:
                current_tokens.append(word)
            else:
                # Stray I-CHARACTER; treat as a new span.
                current_tokens = [word]

        else:
            if current_tokens:
                characters.append(" ".join(current_tokens))
                current_tokens = []

    if current_tokens:
        characters.append(" ".join(current_tokens))

    return characters


def predict(text: str, model_dir: Path, max_length: int = 128):
    id2label = load_id2label(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    words = text.strip().lower().split()

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    word_ids = encoding.word_ids(batch_index=0)

    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    prediction_ids = outputs.logits.argmax(dim=-1)[0].cpu().tolist()

    word_labels = ["O"] * len(words)
    seen_word_ids = set()

    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        # Only use the first sub-token prediction for each original word.
        if word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)
        word_labels[word_id] = id2label[int(prediction_ids[token_index])]

    characters = merge_character_spans(words, word_labels)

    return {
        "words": words,
        "labels": word_labels,
        "characters": characters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/emu-character-ner")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    result = predict(
        text=args.text,
        model_dir=Path(args.model_dir),
        max_length=args.max_length,
    )

    print("\nToken labels:")
    for word, label in zip(result["words"], result["labels"]):
        print(f"{word:20s} {label}")

    print("\nDetected characters:")
    for character in result["characters"]:
        print(f"- {character}")


if __name__ == "__main__":
    main()
# Emu Character BIO Test Dataset

Synthetic dataset for testing a token-classification / NER training pipeline.

## Purpose

This is **not** production-quality training data. It is only meant to let you test the full process:

1. Load token-labelled data
2. Fine-tune a BERT-style token classifier
3. Predict BIO labels
4. Merge `B-CHARACTER` + `I-CHARACTER` spans back into character names

## Labels

- `O`: not a character token
- `B-CHARACTER`: first token of a character name
- `I-CHARACTER`: later token inside the same character name

## Files

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `train.conll`
- `validation.conll`
- `test.conll`
- `label2id.json`
- `id2label.json`

## JSONL format

Each line:

```json
{
  "tokens": ["logen", "ninefingers", "opened", "the", "door"],
  "labels": ["B-CHARACTER", "I-CHARACTER", "O", "O", "O"],
  "ner_tags": [1, 2, 0, 0, 0]
}
```

## Notes

The text is intentionally:
- lowercase
- unpunctuated
- short-window based
- STT-ish
- mixed with generic non-character traps like `the guard`, `the prisoner`, `the lord`, and `the captain`

For a real model, replace or augment this with manually checked audiobook/STT examples.

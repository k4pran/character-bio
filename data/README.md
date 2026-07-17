# Character, Location, and Quote BIO Dataset

CoNLL data for training and testing the repository's token-classification / NER pipeline.

## Purpose

The repository mixes literary, synthetic, and open third-party NER examples to test the full process:

1. Load token-labelled data
2. Fine-tune a BERT-style token classifier
3. Predict BIO labels
4. Merge BIO spans back into characters and named locations

## Labels

- `O`: not part of a labelled entity
- `B-CHARACTER`: first token of a character name
- `I-CHARACTER`: later token inside the same character name
- `B-LOCATION`: first token of a named location
- `I-LOCATION`: later token inside the same named location
- `B-QUOTE`: first token of a quoted span
- `I-QUOTE`: later token inside the same quoted span

## Directories

- `litbank/`: literary examples
- `synthetic/`: generated character/location/quote examples
- `locations/`: normalized Few-NERD, WikiGold, and WNUT 2017 location data
- `label2id.json` and `id2label.json`: the shared label maps

## CoNLL format

Each non-comment line contains a token and BIO label; blank lines separate examples:

```text
new      B-LOCATION
york     I-LOCATION
waited   O
```

## Notes

Much of the project-specific text is intentionally:
- lowercase
- unpunctuated
- short-window based
- STT-ish
- mixed with generic non-character traps like `the guard`, `the prisoner`, `the lord`, and `the captain`

For a real model, replace or augment this with manually checked audiobook/STT examples.

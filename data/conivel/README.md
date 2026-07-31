# Conivel Literary Named-Entity Corpus

This directory contains a project-compatible conversion of the English
literary NER dataset distributed with
[Conivel](https://github.com/CompNet/conivel). The source is the corrected and
expanded version of the OWTO corpus used by Amalvy, Labatut, and Dufour for
research on NER context in novels.

## Why it fits this project

- All 40 documents are first chapters from novels.
- Every book contains both named `PER` and `LOC` annotations.
- The annotation scheme follows CoNLL-style named entities rather than
  LitBank's broader common-noun and pronoun mentions.
- The upstream files are document-level token BIO CoNLL; the importer applies
  the source project's punctuation-based sentence segmentation.

The project conversion maps:

```text
B-PER / I-PER  -> B-CHARACTER / I-CHARACTER
B-LOC / I-LOC  -> B-LOCATION / I-LOCATION
B-ORG / I-ORG  -> O
```

Organization annotations are mapped to `O` because they are outside this
project's label set.

## Provenance and license

- Source repository: <https://github.com/CompNet/conivel>
- Pinned revision: `e8f4ea5112a9a60872f0a726079f8c0f94490b11`
- Upstream dataset directory:
  <https://github.com/CompNet/conivel/tree/e8f4ea5112a9a60872f0a726079f8c0f94490b11/conivel/datas/dekker/dataset>
- Dataset license: Apache License 2.0; see `LICENSE.txt`
- Corpus paper:
  [The Role of Global and Local Context in Named Entity Recognition](https://aclanthology.org/2023.acl-short.62/)
- Original OWTO paper:
  [Evaluating named entity recognition tools for extracting social networks from novels](https://doi.org/10.7717/peerj-cs.189)

Some included novels remain copyrighted in some jurisdictions. The upstream
research dataset distributes only annotated first chapters and labels the
dataset Apache-2.0. Users remain responsible for checking whether their use and
distribution of the underlying excerpts is permitted in their jurisdiction.

## Rebuild

From the repository root:

```powershell
python src/build_conivel_corpus.py
```

The importer downloads a fixed source revision, verifies its SHA-256 checksum,
writes one `.conll` file per book, copies the upstream license, and generates
`manifest.json` with per-book counts and output checksums. It also repairs a
small number of upstream `I-*` labels that start spans so the generated files
are strict BIO.

By default, exact token-sequence overlaps with `data/litbank/` are removed
case-insensitively. This prevents identical book sentences with different
annotation decisions from giving the model contradictory labels. Use
`--keep-overlaps` to reproduce every upstream sentence instead.

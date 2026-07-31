"""Build a reproducible, location-only BIO corpus from open NER datasets.

Non-location entity annotations are intentionally mapped to ``O``. This keeps
the source corpora compatible with this project's CHARACTER/LOCATION label
space without treating real-world people as fictional characters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


WIKIGOLD_URL = (
    "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/"
    "8522f4b39cf9ed323b77f2d8a5005628f8861908/"
    "data/wikigold/CONLL-format/data/wikigold.conll.txt"
)
FEWNERD_REVISION = "205f3e9c9f3577ea2561d43f2f62dc249ab92d5b"
WNUT17_REVISION = "e52a2d2a71ac1a051ca2d532eef28653c5c02603"
WNUT17_URLS = {
    "train": (
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/"
        f"{WNUT17_REVISION}/wnut17train.conll"
    ),
    "validation": (
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/"
        f"{WNUT17_REVISION}/emerging.dev.conll"
    ),
    "test": (
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/"
        f"{WNUT17_REVISION}/emerging.test.annotated"
    ),
}


SOURCE_INFO = {
    "fewnerd": {
        "display_name": "Few-NERD",
        "license": "CC BY-SA 4.0",
        "url": "https://github.com/thunlp/Few-NERD",
        "data_revision": FEWNERD_REVISION,
    },
    "wikigold": {
        "display_name": "WikiGold",
        "license": "CC BY 4.0",
        "url": "https://aclanthology.org/W09-3302/",
        "data_revision": "8522f4b39cf9ed323b77f2d8a5005628f8861908",
    },
    "wnut17": {
        "display_name": "WNUT 2017 Emerging Entities",
        "license": "CC BY 4.0",
        "url": "https://github.com/leondz/emerging_entities_17",
        "data_revision": WNUT17_REVISION,
    },
}


@dataclass(frozen=True)
class Example:
    tokens: tuple[str, ...]
    labels: tuple[str, ...]
    split: str
    location_types: tuple[str, ...] = ()

    @property
    def has_location(self) -> bool:
        return "B-LOCATION" in self.labels


def _entity_part(label: str) -> str:
    lowered = label.lower()
    if "-" in lowered and lowered.split("-", 1)[0] in {"b", "i", "e", "s"}:
        return lowered.split("-", 1)[1]
    return lowered


def is_location_label(label: str) -> bool:
    """Recognize the location labels used by all selected source corpora."""
    entity = _entity_part(label)
    return (
        entity in {"loc", "location", "place"}
        or entity.startswith("location-")
    )


def normalize_location_bio(raw_labels: Sequence[str]) -> list[str]:
    """Map source labels to strict B/I-LOCATION labels and everything else to O."""
    normalized: list[str] = []
    previous_was_location = False
    previous_raw_label = ""

    for raw_label in raw_labels:
        if not is_location_label(raw_label):
            normalized.append("O")
            previous_was_location = False
            previous_raw_label = raw_label
            continue

        prefix = raw_label.lower().split("-", 1)[0]
        has_explicit_prefix = prefix in {"b", "i", "e", "s"}

        starts_new_span = (
            not previous_was_location
            or prefix in {"b", "s"}
            or (not has_explicit_prefix and raw_label != previous_raw_label)
        )
        normalized.append("B-LOCATION" if starts_new_span else "I-LOCATION")
        previous_was_location = prefix not in {"e", "s"}
        previous_raw_label = raw_label

    return normalized


def parse_conll_text(text: str, split: str) -> list[Example]:
    """Parse two-column CoNLL data and normalize its location labels."""
    examples: list[Example] = []
    tokens: list[str] = []
    raw_labels: list[str] = []

    def flush() -> None:
        nonlocal tokens, raw_labels
        if tokens:
            labels = normalize_location_bio(raw_labels)
            location_types = tuple(
                sorted({_entity_part(label) for label in raw_labels if is_location_label(label)})
            )
            examples.append(
                Example(tuple(tokens), tuple(labels), split, location_types)
            )
        tokens = []
        raw_labels = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#") or line.startswith("-DOCSTART-"):
            continue

        parts = line.split()
        if len(parts) < 2:
            raise ValueError(
                f"Malformed CoNLL input at line {line_number}: {raw_line!r}"
            )
        tokens.append(parts[0])
        raw_labels.append(parts[-1])

    flush()
    return examples


def _download_text(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "character-bio-location-corpus/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        text = response.read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return text


def load_wikigold(cache_dir: Path) -> list[Example]:
    text = _download_text(WIKIGOLD_URL, cache_dir / "wikigold.conll.txt")
    return parse_conll_text(text, split="all")


def load_wnut17(cache_dir: Path) -> list[Example]:
    examples: list[Example] = []
    for split, url in WNUT17_URLS.items():
        text = _download_text(url, cache_dir / f"wnut17-{split}.conll")
        examples.extend(parse_conll_text(text, split=split))
    return examples


def _class_label_names(feature) -> list[str]:
    inner_feature = getattr(feature, "feature", feature)
    names = getattr(inner_feature, "names", None)
    if not names:
        raise ValueError(f"Expected a ClassLabel feature, got {feature!r}")
    return list(names)


def load_fewnerd(cache_dir: Path) -> list[Example]:
    # Keep Hugging Face's repository cache with the rest of the reproducible
    # build cache instead of writing to the user's global cache by default.
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface_home" / "hub"))
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Few-NERD import requires the project's 'datasets' dependency. "
            "Install requirements.txt first."
        ) from exc

    dataset = load_dataset(
        "DFKI-SLT/few-nerd",
        "supervised",
        revision=FEWNERD_REVISION,
        cache_dir=str(cache_dir / "huggingface"),
    )
    examples: list[Example] = []

    for split, rows in dataset.items():
        label_names = _class_label_names(rows.features["fine_ner_tags"])
        for row in rows:
            tokens = tuple(str(token) for token in row["tokens"])
            raw_labels = [label_names[int(tag)] for tag in row["fine_ner_tags"]]
            labels = tuple(normalize_location_bio(raw_labels))
            location_types = tuple(
                sorted({_entity_part(label) for label in raw_labels if is_location_label(label)})
            )
            examples.append(Example(tokens, labels, split, location_types))

    return examples


LOADERS = {
    "fewnerd": load_fewnerd,
    "wikigold": load_wikigold,
    "wnut17": load_wnut17,
}


def select_examples(
    examples: Iterable[Example],
    max_positive: int,
    negative_ratio: float,
    seed: int,
) -> list[Example]:
    """Select all/sampled positives plus deterministic hard-negative context."""
    positives: list[Example] = []
    negatives: list[Example] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    for example in examples:
        key = (example.tokens, example.labels)
        if not example.tokens or len(example.tokens) != len(example.labels) or key in seen:
            continue
        seen.add(key)
        (positives if example.has_location else negatives).append(example)

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if max_positive > 0:
        positives = positives[:max_positive]

    negative_count = min(len(negatives), round(len(positives) * negative_ratio))
    selected = positives + negatives[:negative_count]
    rng.shuffle(selected)
    return selected


def _comment_value(value: str) -> str:
    return re.sub(r"[\r\n]+", " ", value).strip()


def write_conll(path: Path, source: str, examples: Sequence[Example]) -> None:
    info = SOURCE_INFO[source]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as output:
        for index, example in enumerate(examples, start=1):
            output.write(f"# id = location-{source}-{index:06d}\n")
            output.write(f"# source = {info['display_name']}\n")
            output.write(f"# source_split = {example.split}\n")
            output.write(f"# license = {info['license']}\n")
            output.write(f"# source_url = {info['url']}\n")
            if example.location_types:
                output.write(f"# location_types = {','.join(example.location_types)}\n")
            output.write(f"# text = {_comment_value(' '.join(example.tokens))}\n")
            for token, label in zip(example.tokens, example.labels):
                safe_token = re.sub(r"\s+", "_", token.strip())
                if not safe_token:
                    raise ValueError(f"Empty token in {source} example {index}")
                output.write(f"{safe_token}\t{label}\n")
            output.write("\n")


def validate_examples(examples: Sequence[Example], source: str) -> None:
    if not examples:
        raise ValueError(f"No examples selected for {source}")
    if not any(example.has_location for example in examples):
        raise ValueError(f"No location spans selected for {source}")

    allowed = {"O", "B-LOCATION", "I-LOCATION"}
    for index, example in enumerate(examples, start=1):
        if len(example.tokens) != len(example.labels):
            raise ValueError(f"Token/label mismatch in {source} example {index}")
        unknown = set(example.labels) - allowed
        if unknown:
            raise ValueError(f"Unknown labels in {source} example {index}: {unknown}")
        previous = "O"
        for label in example.labels:
            if label == "I-LOCATION" and previous not in {"B-LOCATION", "I-LOCATION"}:
                raise ValueError(f"Stray I-LOCATION in {source} example {index}")
            previous = label


def build_corpus(
    sources: Sequence[str],
    output_dir: Path,
    cache_dir: Path,
    max_positive_per_source: int,
    negative_ratio: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}

    for source_number, source in enumerate(sources):
        print(f"Loading {SOURCE_INFO[source]['display_name']}...")
        examples = LOADERS[source](cache_dir)
        selected = select_examples(
            examples,
            max_positive=max_positive_per_source,
            negative_ratio=negative_ratio,
            seed=seed + source_number,
        )
        validate_examples(selected, source)
        output_path = output_dir / f"{source}.conll"
        write_conll(output_path, source, selected)

        location_spans = sum(
            example.labels.count("B-LOCATION") for example in selected
        )
        location_type_examples = Counter(
            location_type
            for example in selected
            for location_type in example.location_types
        )
        summary[source] = {
            "examples": len(selected),
            "tokens": sum(len(example.tokens) for example in selected),
            "location_spans": location_spans,
            "location_type_example_counts": dict(sorted(location_type_examples.items())),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
        print(
            f"Wrote {output_path}: {len(selected)} examples, "
            f"{summary[source]['tokens']} tokens, {location_spans} location spans"
        )

    manifest = {
        "format": "character-bio location corpus manifest v1",
        "build": {
            "seed": seed,
            "max_positive_per_source": max_positive_per_source,
            "negative_ratio": negative_ratio,
        },
        "sources": {
            source: {
                **SOURCE_INFO[source],
                "file": f"{source}.conll",
                **summary[source],
            }
            for source in sources
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(LOADERS),
        default=list(LOADERS),
        help="Source corpora to include (default: all selected corpora).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/locations"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/location_corpus"),
    )
    parser.add_argument(
        "--max-positive-per-source",
        type=int,
        default=4000,
        help="Maximum location-positive examples per source; 0 keeps all.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.25,
        help="Negative-only examples per selected positive example.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.negative_ratio < 0:
        parser.error("--negative-ratio must be non-negative")
    if args.max_positive_per_source < 0:
        parser.error("--max-positive-per-source must be non-negative")

    build_corpus(
        sources=args.sources,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_positive_per_source=args.max_positive_per_source,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

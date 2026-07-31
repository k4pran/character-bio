"""Build project-compatible BIO CoNLL files from the Conivel literary corpus.

Conivel's corrected OWTO dataset contains first chapters from 40 English
novels annotated with named PER, LOC, and ORG entities. This importer maps PER
to CHARACTER, LOC to LOCATION, and unsupported ORG labels to O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SOURCE_REVISION = "e8f4ea5112a9a60872f0a726079f8c0f94490b11"
SOURCE_ARCHIVE_URL = (
    "https://github.com/CompNet/conivel/archive/"
    f"{SOURCE_REVISION}.zip"
)
SOURCE_ARCHIVE_SHA256 = (
    "c023b7e41d3f77d8f933ae71a3d5df482061ab1adaf5f6b83d62fdb1ae8967e3"
)
SOURCE_REPOSITORY_URL = "https://github.com/CompNet/conivel"
SOURCE_DATASET_URL = (
    f"{SOURCE_REPOSITORY_URL}/tree/{SOURCE_REVISION}/"
    "conivel/datas/dekker/dataset"
)
SOURCE_LICENSE = "Apache-2.0"

SOURCE_LABELS = {
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
}
LABEL_MAP = {
    "O": "O",
    "B-PER": "B-CHARACTER",
    "I-PER": "I-CHARACTER",
    "B-LOC": "B-LOCATION",
    "I-LOC": "I-LOCATION",
    "B-ORG": "O",
    "I-ORG": "O",
}
PROJECT_LABELS = {
    "O",
    "B-CHARACTER",
    "I-CHARACTER",
    "B-LOCATION",
    "I-LOCATION",
}

BOOK_TITLES = {
    "1984": "1984",
    "AGameOfThrones": "A Game of Thrones",
    "AliceInWonderland": "Alice in Wonderland",
    "AssassinsApprentice": "Assassin's Apprentice",
    "AStudyInScarlet": "A Study in Scarlet",
    "BlackPrism": "The Black Prism",
    "BraveNewWorld": "Brave New World",
    "DavidCopperfield": "David Copperfield",
    "Dracula": "Dracula",
    "Elantris": "Elantris",
    "Emma": "Emma",
    "Frankenstein": "Frankenstein",
    "GardensOfTheMoon": "Gardens of the Moon",
    "HarryPotter": "Harry Potter and the Philosopher's Stone",
    "HuckleberryFinn": "Adventures of Huckleberry Finn",
    "JekyllAndHyde": "Strange Case of Dr Jekyll and Mr Hyde",
    "Magician": "Magician",
    "Mistborn": "Mistborn",
    "MobyDick": "Moby-Dick",
    "OliverTwist": "Oliver Twist",
    "PrideAndPrejudice": "Pride and Prejudice",
    "StormFront": "Storm Front",
    "TheBlackCompany": "The Black Company",
    "TheBladeItself": "The Blade Itself",
    "TheCallOfTheWild": "The Call of the Wild",
    "TheColourOfMagic": "The Colour of Magic",
    "TheCountOfMonteCristo": "The Count of Monte Cristo",
    "TheFellowshipoftheRing": "The Fellowship of the Ring",
    "TheGunslinger": "The Gunslinger",
    "TheLiesOfLockeLamora": "The Lies of Locke Lamora",
    "TheNameOfTheWind": "The Name of the Wind",
    "ThePaintedMan": "The Painted Man",
    "TheThreeMusketeers": "The Three Musketeers",
    "TheWayOfKings": "The Way of Kings",
    "TheWayOfShadows": "The Way of Shadows",
    "TheWayWeLiveNow": "The Way We Live Now",
    "TheWheelOfTime": "The Eye of the World",
    "TinkerTailorSoldierSpy": "Tinker Tailor Soldier Spy",
    "Ulysses": "Ulysses",
    "VanityFair": "Vanity Fair",
}


@dataclass(frozen=True)
class Sentence:
    tokens: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def character_spans(self) -> int:
        return self.labels.count("B-CHARACTER")

    @property
    def location_spans(self) -> int:
        return self.labels.count("B-LOCATION")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def filename_slug(source_stem: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", source_stem)
    return re.sub(r"[^a-zA-Z0-9]+", "_", spaced).strip("_").lower()


def normalize_labels(raw_labels: Sequence[str]) -> tuple[str, ...]:
    unknown = set(raw_labels) - SOURCE_LABELS
    if unknown:
        raise ValueError(f"Unknown Conivel labels: {sorted(unknown)}")

    normalized: list[str] = []
    previous = "O"
    for raw_label in raw_labels:
        label = LABEL_MAP[raw_label]
        if label.startswith("I-"):
            entity_type = label[2:]
            if previous not in {f"B-{entity_type}", f"I-{entity_type}"}:
                label = f"B-{entity_type}"
        normalized.append(label)
        previous = label
    return tuple(normalized)


def validate_sentences(sentences: Sequence[Sentence], source: str) -> None:
    if not sentences:
        raise ValueError(f"No sentences found in {source}")

    for sentence_number, sentence in enumerate(sentences, start=1):
        if not sentence.tokens or len(sentence.tokens) != len(sentence.labels):
            raise ValueError(
                f"Token/label mismatch in {source} sentence {sentence_number}"
            )
        unknown = set(sentence.labels) - PROJECT_LABELS
        if unknown:
            raise ValueError(
                f"Unknown project labels in {source} sentence "
                f"{sentence_number}: {sorted(unknown)}"
            )

        previous = "O"
        for label in sentence.labels:
            if label == "I-CHARACTER" and previous not in {
                "B-CHARACTER",
                "I-CHARACTER",
            }:
                raise ValueError(
                    f"Stray I-CHARACTER in {source} sentence {sentence_number}"
                )
            if label == "I-LOCATION" and previous not in {
                "B-LOCATION",
                "I-LOCATION",
            }:
                raise ValueError(
                    f"Stray I-LOCATION in {source} sentence {sentence_number}"
                )
            previous = label


def sentence_key(tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(token.casefold() for token in tokens)


def load_project_sentence_keys(directory: Path) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for path in sorted(directory.rglob("*.conll")):
        tokens: list[str] = []

        def flush() -> None:
            nonlocal tokens
            if tokens:
                keys.add(sentence_key(tokens))
            tokens = []

        with path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    flush()
                    continue
                if line.startswith("#") and not line.startswith("#\t"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"{path}:{line_number} expected token and label"
                    )
                tokens.append(parts[0])
        flush()
    return keys


def filter_overlaps(
    sentences: Sequence[Sentence],
    existing_keys: set[tuple[str, ...]],
) -> tuple[list[Sentence], int]:
    kept = [
        sentence
        for sentence in sentences
        if sentence_key(sentence.tokens) not in existing_keys
    ]
    return kept, len(sentences) - len(kept)


def parse_conll(path: Path) -> list[Sentence]:
    sentences: list[Sentence] = []
    rows: list[tuple[str, str] | None] = []
    tokens: list[str] = []
    raw_labels: list[str] = []

    def flush() -> None:
        nonlocal tokens, raw_labels
        if tokens:
            sentences.append(
                Sentence(tuple(tokens), normalize_labels(raw_labels))
            )
        tokens = []
        raw_labels = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                rows.append(None)
                continue

            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_number} expected token and label: {line!r}"
                )
            token, label = parts
            rows.append((token, label))

    for row_number, row in enumerate(rows):
        if row is None:
            flush()
            continue

        token, label = row
        tokens.append(token)
        raw_labels.append(label)

        next_row = rows[row_number + 1] if row_number + 1 < len(rows) else None
        next_token = next_row[0] if next_row is not None else None

        # Match Conivel's own loader: punctuation immediately before a closing
        # quote stays with that quote; otherwise punctuation closes a sentence.
        if next_token == "''":
            continue
        if token in {"''", ".", "?", "!"}:
            flush()

    flush()
    validate_sentences(sentences, str(path))
    return sentences


def _safe_comment(value: str) -> str:
    return re.sub(r"[\r\n]+", " ", value).strip()


def write_book(
    output_path: Path,
    source_filename: str,
    title: str,
    sentences: Sequence[Sentence],
) -> dict[str, int | str]:
    validate_sentences(sentences, source_filename)
    book_id = slugify(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for sentence_number, sentence in enumerate(sentences, start=1):
            output.write(
                f"# id = conivel-{book_id}-{sentence_number:06d}\n"
            )
            output.write("# source = Conivel corrected OWTO literary NER\n")
            output.write(f"# book = {_safe_comment(title)}\n")
            output.write(f"# source_file = {source_filename}\n")
            output.write(f"# source_revision = {SOURCE_REVISION}\n")
            output.write(f"# license = {SOURCE_LICENSE}\n")
            output.write(f"# source_url = {SOURCE_DATASET_URL}\n")
            output.write(
                f"# text = {_safe_comment(' '.join(sentence.tokens))}\n"
            )
            for token, label in zip(sentence.tokens, sentence.labels):
                safe_token = re.sub(r"\s+", "_", token.strip())
                if not safe_token:
                    raise ValueError(
                        f"Empty token in {source_filename} sentence "
                        f"{sentence_number}"
                    )
                output.write(f"{safe_token}\t{label}\n")
            output.write("\n")

    return {
        "sentences": len(sentences),
        "tokens": sum(len(sentence.tokens) for sentence in sentences),
        "character_spans": sum(
            sentence.character_spans for sentence in sentences
        ),
        "location_spans": sum(
            sentence.location_spans for sentence in sentences
        ),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def _archive_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_ARCHIVE_SHA256


def _download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SOURCE_ARCHIVE_URL,
        headers={"User-Agent": "character-bio-conivel-importer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            f"Conivel archive checksum mismatch: expected "
            f"{SOURCE_ARCHIVE_SHA256}, got {digest}"
        )
    path.write_bytes(data)


def _extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    expected_dataset = (
        extract_dir
        / f"conivel-{SOURCE_REVISION}"
        / "conivel"
        / "datas"
        / "dekker"
        / "dataset"
    )
    if expected_dataset.exists():
        return expected_dataset

    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = extract_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if target != resolved_root and resolved_root not in target.parents:
                raise ValueError(
                    f"Unsafe path in Conivel archive: {member.filename}"
                )
        archive.extractall(extract_dir)

    if not expected_dataset.exists():
        raise FileNotFoundError(
            f"Conivel dataset missing after extraction: {expected_dataset}"
        )
    return expected_dataset


def resolve_source_dataset(
    cache_dir: Path,
    source_dir: Path | None = None,
) -> Path:
    if source_dir is not None:
        candidates = (
            source_dir,
            source_dir / "conivel" / "datas" / "dekker" / "dataset",
        )
        for candidate in candidates:
            if candidate.exists() and list(candidate.glob("*.conll")):
                return candidate
        raise FileNotFoundError(
            f"Could not find Conivel .conll files under {source_dir}"
        )

    archive_path = cache_dir / f"conivel-{SOURCE_REVISION}.zip"
    if not _archive_is_valid(archive_path):
        _download_archive(archive_path)
    return _extract_archive(archive_path, cache_dir / "source")


def build_corpus(
    output_dir: Path,
    cache_dir: Path,
    source_dir: Path | None = None,
    dedupe_against: Path | None = Path("data/litbank"),
) -> dict[str, dict[str, int | str]]:
    dataset_dir = resolve_source_dataset(cache_dir, source_dir)
    source_files = sorted(dataset_dir.glob("*.conll"))
    source_stems = {path.stem for path in source_files}
    existing_keys = (
        load_project_sentence_keys(dedupe_against)
        if dedupe_against is not None and dedupe_against.exists()
        else set()
    )

    missing = set(BOOK_TITLES) - source_stems
    unexpected = source_stems - set(BOOK_TITLES)
    if missing or unexpected:
        raise ValueError(
            f"Unexpected Conivel book set; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int | str]] = {}
    totals: Counter[str] = Counter()

    for source_path in source_files:
        title = BOOK_TITLES[source_path.stem]
        output_name = f"{filename_slug(source_path.stem)}.conll"
        output_path = output_dir / output_name
        source_sentences = parse_conll(source_path)
        sentences, overlaps_removed = filter_overlaps(
            source_sentences,
            existing_keys,
        )
        stats = write_book(
            output_path,
            source_filename=source_path.name,
            title=title,
            sentences=sentences,
        )
        if not stats["character_spans"] or not stats["location_spans"]:
            raise ValueError(
                f"{source_path.name} does not contain both required entity types"
            )
        summary[source_path.stem] = {
            "book": title,
            "file": output_name,
            "source_sentences": len(source_sentences),
            "overlap_sentences_removed": overlaps_removed,
            **stats,
        }
        totals.update(
            {
                key: int(stats[key])
                for key in (
                    "sentences",
                    "tokens",
                    "character_spans",
                    "location_spans",
                )
            }
        )
        totals["source_sentences"] += len(source_sentences)
        totals["overlap_sentences_removed"] += overlaps_removed
        print(
            f"Wrote {output_path}: {stats['sentences']} sentences, "
            f"{stats['character_spans']} character spans, "
            f"{stats['location_spans']} location spans, "
            f"{overlaps_removed} overlaps removed"
        )

    source_license = dataset_dir / "LICENSE.txt"
    if not source_license.exists():
        raise FileNotFoundError(f"Missing upstream license: {source_license}")
    shutil.copyfile(source_license, output_dir / "LICENSE.txt")

    manifest = {
        "format": "character-bio Conivel corpus manifest v1",
        "source": {
            "name": "Conivel corrected OWTO literary NER",
            "repository": SOURCE_REPOSITORY_URL,
            "dataset_url": SOURCE_DATASET_URL,
            "revision": SOURCE_REVISION,
            "archive_url": SOURCE_ARCHIVE_URL,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "license": SOURCE_LICENSE,
            "label_mapping": LABEL_MAP,
        },
        "deduplication": {
            "against": (
                str(dedupe_against).replace("\\", "/")
                if dedupe_against is not None
                else None
            ),
            "comparison": "exact token sequence after Unicode case-folding",
        },
        "totals": dict(sorted(totals.items())),
        "books": summary,
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/conivel"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/conivel_corpus"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Optional existing Conivel checkout or dataset directory. "
            "When omitted, the pinned source archive is downloaded."
        ),
    )
    parser.add_argument(
        "--dedupe-against",
        type=Path,
        default=Path("data/litbank"),
        help=(
            "Remove exact case-insensitive sentence overlaps with this "
            "project CoNLL directory (default: data/litbank)."
        ),
    )
    parser.add_argument(
        "--keep-overlaps",
        action="store_true",
        help="Keep sentences that duplicate the existing literary corpus.",
    )
    args = parser.parse_args()
    build_corpus(
        args.output_dir,
        args.cache_dir,
        args.source_dir,
        None if args.keep_overlaps else args.dedupe_against,
    )


if __name__ == "__main__":
    main()

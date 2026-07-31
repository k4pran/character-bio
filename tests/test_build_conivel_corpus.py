import tempfile
import unittest
from pathlib import Path

from src.build_conivel_corpus import (
    Sentence,
    filter_overlaps,
    normalize_labels,
    parse_conll,
    validate_sentences,
    write_book,
)


class ConivelCorpusTests(unittest.TestCase):
    def test_maps_person_and_location_to_project_labels(self):
        self.assertEqual(
            normalize_labels(
                ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "B-ORG"]
            ),
            (
                "B-CHARACTER",
                "I-CHARACTER",
                "O",
                "B-LOCATION",
                "I-LOCATION",
                "O",
            ),
        )

    def test_rejects_unknown_source_labels(self):
        with self.assertRaisesRegex(ValueError, "Unknown Conivel labels"):
            normalize_labels(["B-FAC"])

    def test_repairs_stray_inside_labels_from_upstream(self):
        self.assertEqual(
            normalize_labels(["O", "I-PER", "O", "I-LOC"]),
            ("O", "B-CHARACTER", "O", "B-LOCATION"),
        )

    def test_parses_and_validates_conll_sentences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "book.conll"
            path.write_text(
                "Mr. B-PER\nBennet I-PER\nvisited O\n"
                "Netherfield B-LOC\nPark I-LOC\n\n"
                "He O\nleft O\n. O\n",
                encoding="utf-8",
            )
            sentences = parse_conll(path)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].character_spans, 1)
        self.assertEqual(sentences[0].location_spans, 1)
        self.assertEqual(sentences[1].labels, ("O", "O", "O"))

    def test_rejects_stray_inside_labels(self):
        with self.assertRaisesRegex(ValueError, "Stray I-CHARACTER"):
            validate_sentences(
                [Sentence(("Bennet",), ("I-CHARACTER",))],
                "fixture.conll",
            )

    def test_filters_case_insensitive_sentence_overlaps(self):
        duplicate = Sentence(("Mr.", "Bennet"), ("B-CHARACTER", "I-CHARACTER"))
        unique = Sentence(("Netherfield",), ("B-LOCATION",))
        kept, removed = filter_overlaps(
            [duplicate, unique],
            {("mr.", "bennet")},
        )
        self.assertEqual(kept, [unique])
        self.assertEqual(removed, 1)

    def test_writes_project_conll_with_provenance(self):
        sentences = [
            Sentence(
                ("Mr.", "Bennet", "visited", "Netherfield"),
                ("B-CHARACTER", "I-CHARACTER", "O", "B-LOCATION"),
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pride_and_prejudice.conll"
            stats = write_book(
                path,
                "PrideAndPrejudice.conll",
                "Pride and Prejudice",
                sentences,
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("Mr.\tB-CHARACTER", text)
        self.assertIn("Netherfield\tB-LOCATION", text)
        self.assertIn("# license = Apache-2.0", text)
        self.assertEqual(stats["character_spans"], 1)
        self.assertEqual(stats["location_spans"], 1)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from src.build_location_corpus import (
    Example,
    normalize_location_bio,
    parse_conll_text,
    select_examples,
    validate_examples,
    write_conll,
)


class LocationCorpusTests(unittest.TestCase):
    def test_normalizes_bio_and_io_schemas(self):
        self.assertEqual(
            normalize_location_bio(
                ["O", "B-location", "I-location", "O", "location-GPE", "location-GPE"]
            ),
            ["O", "B-LOCATION", "I-LOCATION", "O", "B-LOCATION", "I-LOCATION"],
        )

    def test_maps_non_location_entities_to_o(self):
        self.assertEqual(
            normalize_location_bio(["B-person", "I-person", "B-organization"]),
            ["O", "O", "O"],
        )

    def test_parses_conll_and_repairs_stray_i(self):
        examples = parse_conll_text(
            "London I-LOC\nis O\nhere O\n\nAlice B-PER\nleft O\n",
            split="train",
        )
        self.assertEqual(examples[0].labels, ("B-LOCATION", "O", "O"))
        self.assertFalse(examples[1].has_location)

    def test_selection_is_deterministic_and_deduplicated(self):
        positive = Example(("London",), ("B-LOCATION",), "train")
        negative = Example(("Alice",), ("O",), "train")
        examples = [positive, positive, negative]
        first = select_examples(examples, 10, 1.0, seed=7)
        second = select_examples(examples, 10, 1.0, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_written_file_is_valid_project_conll(self):
        examples = [
            Example(
                ("New", "York", "welcomed", "Alice"),
                ("B-LOCATION", "I-LOCATION", "O", "O"),
                "train",
            )
        ]
        validate_examples(examples, "wikigold")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "wikigold.conll"
            write_conll(path, "wikigold", examples)
            text = path.read_text(encoding="utf-8")
        self.assertIn("New\tB-LOCATION", text)
        self.assertIn("York\tI-LOCATION", text)
        self.assertIn("# license = CC BY 4.0", text)


if __name__ == "__main__":
    unittest.main()

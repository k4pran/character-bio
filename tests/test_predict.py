import unittest

from src.predict import merge_entity_spans


class PredictSpanTests(unittest.TestCase):
    def test_merges_location_spans(self):
        words = ["from", "new", "york", "to", "paris"]
        labels = ["O", "B-LOCATION", "I-LOCATION", "O", "B-LOCATION"]
        self.assertEqual(
            merge_entity_spans(words, labels, "LOCATION"),
            ["new york", "paris"],
        )

    def test_other_entity_types_close_current_span(self):
        words = ["london", "alice"]
        labels = ["B-LOCATION", "B-CHARACTER"]
        self.assertEqual(
            merge_entity_spans(words, labels, "LOCATION"),
            ["london"],
        )


if __name__ == "__main__":
    unittest.main()

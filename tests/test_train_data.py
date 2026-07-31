import tempfile
import unittest
from pathlib import Path

from src.train import find_conll_files, load_conll_file


class TrainDataDiscoveryTests(unittest.TestCase):
    def test_ignores_download_cache_conll_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            cache_dir = data_dir / ".cache" / "source"
            cache_dir.mkdir(parents=True)
            training_file = data_dir / "train-source.conll"
            cached_file = cache_dir / "raw-source.conll"
            training_file.write_text("london\tB-LOCATION\n", encoding="utf-8")
            cached_file.write_text("London\tB-location\n", encoding="utf-8")

            self.assertEqual(find_conll_files(data_dir), [training_file])

    def test_loads_literal_hash_token_without_treating_it_as_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hash-token.conll"
            path.write_text(
                "# id = hash-example\n#\tO\nLondon\tB-LOCATION\n",
                encoding="utf-8",
            )
            rows = load_conll_file(
                path,
                {
                    "O": 0,
                    "B-LOCATION": 1,
                },
            )

        self.assertEqual(rows[0]["tokens"], ["#", "London"])
        self.assertEqual(rows[0]["labels"], ["O", "B-LOCATION"])


if __name__ == "__main__":
    unittest.main()

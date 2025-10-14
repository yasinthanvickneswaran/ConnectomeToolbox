import unittest
from pathlib import Path

from cect.ConnectomeDataset import (
    ConnectomeDataset,
    get_cache_filename,
)
from cect.White_whole import get_cache, get_instance


class TestWhiteWhole(unittest.TestCase):
    def setUp(self):
        self.cache_files_to_cleanup = []

    def tearDown(self):
        for file in self.cache_files_to_cleanup:
            Path(file).unlink(missing_ok=True)

    def test_get_instance(self):
        current_filepath = Path(__file__)
        spreadsheet_directory: Path = current_filepath.parents[1] / "data"
        spreadsheet_filepath: Path = (
            spreadsheet_directory / "aconnectome_white_1986_whole.csv"
        )

        assert spreadsheet_filepath.is_file(), (
            f"Test data file should exist at {spreadsheet_filepath}"
        )
        filename = str(spreadsheet_filepath)
        instance = get_instance(from_cache=False, spreadsheet_location=filename)
        data = instance.read_data()

        assert any([any(_) for _ in data]), "Instance should contain data"

    def test_get_cache(self):
        current_filepath = Path(__file__)
        spreadsheet_directory: Path = current_filepath.parents[1] / "data"
        spreadsheet_filepath: Path = (
            spreadsheet_directory / "aconnectome_white_1986_whole.csv"
        )
        filename = str(spreadsheet_filepath)
        white_cache = get_cache_filename("White_whole")
        if not Path(white_cache).is_file():
            self.cache_files_to_cleanup.append(white_cache)
        else:
            print(f"Cache file {white_cache} already exists - not cleaning it up")

        instance = get_instance(from_cache=False, spreadsheet_location=filename)
        instance.save_to_cache("White_whole")

        cache: ConnectomeDataset | None = get_cache()

        assert cache is not None, "Cache should not be None"
        assert any([any(_) for _ in cache._read_data()]), "Cache should contain data"
        assert isinstance(cache, ConnectomeDataset), (
            "Cache should be of type ConnectomeDataset"
        )
        assert instance._read_data() == cache._read_data(), (
            "Data from instance and cache should be identical"
        )


if __name__ == "__main__":
    unittest.main()

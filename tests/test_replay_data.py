import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_data import ReplayCache


class ReplayDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.data = gzip.compress(b'actual recording bytes', mtime=0)
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.name = self.digest + '.gz'
        part = dict(file='data/packed/' + self.name, sha256=self.digest,
                    compressed_bytes=len(self.data), bytes=22)
        self.index = dict(files={'example': dict(parts=[part])})

    def cache(self, offline=False):
        return ReplayCache(self.folder.name, offline=offline, index=self.index,
                           origin='https://example.org')

    def test_missing_host_does_not_attempt_network(self):
        with patch.dict('os.environ', {}, clear=True), patch('replay_data.urlopen') as fetch:
            cache = ReplayCache(self.folder.name, index=self.index)
            with self.assertRaisesRegex(FileNotFoundError, 'FRUIT_FLY_DATA_URL'):
                cache.get(self.name)
            fetch.assert_not_called()

    def test_data_host_can_be_set_outside_repository(self):
        with patch.dict('os.environ', {'FRUIT_FLY_DATA_URL': 'https://example.org/'}):
            cache = ReplayCache(self.folder.name, index=self.index)
            self.assertEqual(cache.origin, 'https://example.org')

    def test_verified_cache_works_offline(self):
        path = Path(self.folder.name) / self.name
        path.write_bytes(self.data)
        self.assertEqual(self.cache(True).get(self.name), path)

    def test_corruption_and_unknown_paths_fail_closed(self):
        path = Path(self.folder.name) / self.name
        path.write_bytes(b'x' * len(self.data))
        cache = self.cache(True)
        for name in (self.name, '../README.md', 'a' * 64 + '.gz'):
            with self.assertRaises(FileNotFoundError):
                cache.get(name)

    def test_successful_download_is_atomic_and_reused(self):
        cache = self.cache()
        with patch('replay_data.urlopen', return_value=io.BytesIO(self.data)) as fetch:
            self.assertEqual(cache.get(self.name).read_bytes(), self.data)
            cache.get(self.name)
            self.assertEqual(fetch.call_count, 1)
        self.assertFalse(list(Path(self.folder.name).glob('*.part')))

    def test_invalid_download_never_becomes_cache(self):
        for bad in (b'short', b'x' * len(self.data), self.data + b'extra'):
            with patch('replay_data.urlopen', return_value=io.BytesIO(bad)):
                with self.assertRaises(ValueError):
                    self.cache().get(self.name)
            self.assertEqual(list(Path(self.folder.name).iterdir()), [])


if __name__ == '__main__':
    unittest.main()

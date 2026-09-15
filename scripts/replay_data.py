"""Verified, immutable replay downloads. Python standard library only."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


class ReplayCache:
    def __init__(self, directory=None, offline=False, origin=None, index=None):
        self.directory = Path(directory) if directory else ROOT / '.cache/replay'
        self.offline = offline
        self.origin = (origin or os.environ.get('FRUIT_FLY_DATA_URL', '')).rstrip('/')
        if index is None:
            index = json.loads((ROOT / 'viewer/data/pack-index.json').read_text())
        self.parts = {}
        for entry in index['files'].values():
            for part in entry['parts']:
                name = part['file'].removeprefix('data/packed/')
                if not re.fullmatch(r'[a-f0-9]{64}\.gz', name) or name[:-3] != part['sha256']:
                    raise ValueError('Invalid content-addressed replay chunk')
                if name in self.parts and self.parts[name] != part:
                    raise ValueError('Conflicting replay metadata')
                self.parts[name] = part
        self._locks = {name: threading.Lock() for name in self.parts}
        self._verified = {}

    def valid(self, path, part):
        if not path.is_file() or path.stat().st_size != part['compressed_bytes']:
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == part['sha256']

    def get(self, name):
        if name not in self.parts:
            raise FileNotFoundError('Chunk is not part of this release')
        with self._locks[name]:
            part = self.parts[name]
            path = self.directory / name
            stamp = None
            if path.is_file():
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
            if stamp and self._verified.get(name) == stamp:
                return path
            if self.valid(path, part):
                self._verified[name] = stamp
                return path
            if self.offline:
                raise FileNotFoundError('Replay is missing or corrupt; run scripts/fetch_data.py online')
            if not self.origin:
                raise FileNotFoundError('No data host configured. Set FRUIT_FLY_DATA_URL or supply an existing --cache directory.')
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=name, suffix='.part', dir=self.directory)
            try:
                # Bound each response by its exact expected size before caching it.
                with os.fdopen(fd, 'wb') as output, urlopen(
                    Request(self.origin + '/data/packed/' + name,
                            headers={'User-Agent': 'FruitFlySnake/0.1 (public replay downloader)'}), timeout=60
                ) as response:
                    total = 0
                    while block := response.read(min(1024 * 1024, part['compressed_bytes'] - total + 1)):
                        total += len(block)
                        if total > part['compressed_bytes']:
                            raise ValueError('Replay download exceeds its expected size')
                        output.write(block)
                if not self.valid(Path(temporary), part):
                    raise ValueError('Replay download failed its SHA-256 or size check')
                os.replace(temporary, path)
                stat = path.stat()
                self._verified[name] = (stat.st_size, stat.st_mtime_ns)
                return path
            finally:
                Path(temporary).unlink(missing_ok=True)

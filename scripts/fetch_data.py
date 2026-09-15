"""Download and verify all selected replay data for offline use."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from replay_data import ReplayCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    cache = ReplayCache(args.cache)
    size = sum(x['compressed_bytes'] for x in cache.parts.values())
    print(f'{len(cache.parts):,} chunks, {size / 1e6:.1f} MB total; verified cached files are reused.')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(cache.get, cache.parts), 1):
            if i % 100 == 0 or i == len(cache.parts):
                print(f'Verified {i:,}/{len(cache.parts):,}', flush=True)


if __name__ == '__main__':
    main()

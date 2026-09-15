"""Check source provenance, local documentation links, and packaging boundaries."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    inventory = json.loads((ROOT / 'docs/source-inventory.json').read_text())
    for entry in inventory['files']:
        path = ROOT / entry['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            errors.append('Original source changed: ' + entry['file'])
    for path in ROOT.rglob('*.md'):
        if any(x in path.parts for x in ('.git', '.venv', '.cache', 'node_modules')):
            continue
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                if not (path.parent / target.split('#')[0]).exists():
                    errors.append(f'Broken link in {path.relative_to(ROOT)}: {target}')
    if (ROOT / '.git').exists():
        files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        for name in filter(None, files):
            if name.startswith(('.cache/', 'model/data/', 'model/work/')):
                errors.append('Local artifact is tracked: ' + name)
            if (ROOT / name).stat().st_size > 10 * 1024 ** 2:
                errors.append('Unexpected large tracked file: ' + name)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'{len(inventory["files"])} original source hashes, documentation links and packaging checks passed.')


if __name__ == '__main__':
    main()

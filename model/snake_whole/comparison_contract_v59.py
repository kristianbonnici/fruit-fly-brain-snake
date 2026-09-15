"""Shared immutable allocation and lightweight resource checks for V59."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

from .common import ROOT, digest, stable

BASE = ROOT/'data/snake-whole-v59'


def environment():
    return dict(python=sys.version, executable=sys.executable, mlx=importlib.metadata.version('mlx'),
                numpy=importlib.metadata.version('numpy'))


def contract():
    path = BASE/'comparison-contract.json'
    plan = json.loads(path.read_text())
    if plan['status'] != 'allocated_main_comparisons' or plan['environment'] != environment():
        raise ValueError('Frozen main comparison and exact runtime required')
    for p, sha in {**plan['sources'], **plan['evidence']}.items():
        if digest(ROOT/p) != sha:
            raise ValueError('Frozen comparison evidence changed: '+p)
    return plan, digest(path)


def resources(seconds_limit, started, previous=0.):
    rows = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    for line in rows.splitlines():
        m = re.match(r'^\s*(\d+)\s+(\S+)\s+-m\s+(snake_whole\.\S+)', line)
        if m and int(m[1]) != os.getpid() and m[3] != 'snake_whole.viewer' and re.fullmatch(r'Python|python(?:3(?:\.\d+)?)?', Path(m[2]).name):
            raise RuntimeError('Competing neural process: '+line.strip())
    if (previous+time.perf_counter()-started > seconds_limit
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8_000_000_000
            or shutil.disk_usage(BASE).free < 8_000_000_000):
        raise RuntimeError('Frozen process time, memory or disk limit reached')


def run_directory(plan, kind, seed):
    if kind not in ('pixels', 'neural'):
        raise ValueError('Explicit comparison kind required')
    first, second = plan['training']['first_seed'], plan['training']['reproduction_seed']
    if seed != first:
        report = BASE/f'review-{first}.json'
        if seed != second or kind != 'pixels' or not report.exists():
            raise ValueError('The reproduction is conditional on the first image learning gate')
        r = json.loads(report.read_text())
        identity = r.pop('identity')
        if (stable(r) != identity or r['binding'] != digest(BASE/'comparison-contract.json') or not r['image_learning_passed']
                or any(digest(ROOT/p) != sha for p, sha in r['artifact_evidence'].items())):
            raise ValueError('No reproduction allocated after a failed first learning gate')
    return BASE/f'runs/{kind}-{seed}'

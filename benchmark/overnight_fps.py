#!/usr/bin/env python3
"""RISC-V gem5 FPS batch: resumable, indices verified, rows saved after each run."""
from pathlib import Path
import csv
import os
import re
import subprocess
import sys
from datetime import datetime

ROOT = Path.cwd()
DATA = ROOT / 'datasets/kitti/0000000000.bin'
GEM5 = ROOT / 'gem5/build/RISCV/gem5.opt'
CONFIG = ROOT / 'benchmark/cpu/riscv_fps_comparison.py'
BASE = ROOT / 'benchmark/stats/experiments/fps/overnight_20261122'
BASE.mkdir(parents=True, exist_ok=True)
CSV = BASE / 'summary.csv'

# 8-case N sweep at M=128, 8-case M sweep at N=100k,
# plus N=50k and N=16384 M sweeps; 26 distinct combinations.
NS = [1024, 2048, 4096, 10000, 16384, 20000, 50000, 100000]
MS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
CASES = list(dict.fromkeys(
    [(n, 128) for n in NS] +
    [(100000, m) for m in MS] +
    [(50000, m) for m in MS] +
    [(16384, m) for m in [128, 512, 1024, 2048, 4096]]
))
LMUL_CASES = list(dict.fromkeys(
    [(16384, m) for m in [128, 512, 1024, 2048, 4096]] +
    [(100000, m) for m in [128, 512, 1024, 2048, 4096]]
))
BINS = {
    'main_scalar': 'spherical_fps_main_scalar',
    'main_rvv_m1': 'spherical_fps_main_rvv_m1',
    'ref_scalar': 'spherical_fps_ref_scalar',
    'ref_rvv_m1': 'spherical_fps_ref_rvv_m1',
    'radius_scalar': 'radius_fps_scalar',
    'radius_rvv_m1': 'radius_fps_rvv_m1',
    **{f'main_rvv_m{k}': f'spherical_fps_main_rvv_m{k}' for k in (2, 4, 8)},
    # RadiusFPS LMUL 2/4/8 excluded: M2 failed exact correctness.
}
FIELDS = ['name','N','M','status','prep_ms','sample_ms','total_ms','sim_ms','sim_insts','index_count','source','timestamp','note']


def parse_stats(path, split):
    text = path.read_text(errors='replace')
    ticks = [int(x) for x in re.findall(r'^simTicks\s+(\d+)', text, re.M)]
    insts = [int(x) for x in re.findall(r'^simInsts\s+(\d+)', text, re.M)]
    if not ticks:
        raise ValueError('No simTicks in stats.txt')
    if split:
        if len(ticks) < 2:
            raise ValueError('Split statistics expected, but only one stats section found')
        return (ticks[0]+ticks[1])/1e9, sum(insts[:2]) if len(insts)>=2 else ''
    return ticks[0]/1e9, insts[0] if insts else ''


def history_path(name, n, m):
    # Only existing scalar baseline and reference runs are reused.
    if name == 'radius_scalar':
        root = ROOT / 'benchmark/stats/experiments/fps/RadiusFPS/kitti/full_radix_sweep'
        return root / f'N{n}_M{m}' / 'stats.txt'
    if name == 'ref_scalar':
        root = ROOT / 'spherical_fps/ref_implementation/results/spherical_fps'
        if n == 100000 and m == 128:
            return root / 'N100000_M128/stats.txt'
        if n == 100000:
            return root / 'M_sweep' / f'N{n}_M{m}' / 'stats.txt'
        if m == 128:
            return root / 'N_sweep' / f'N{n}_M{m}' / 'stats.txt'
    return None


def parse_output(path):
    s = path.read_text(errors='replace')
    d = {k: v for k, v in re.findall(r'^([A-Z][A-Z0-9_]+)=([^\r\n]+)', s, re.M)}
    pairs = [(int(i), int(v)) for i, v in re.findall(r'^FPS_INDEX\[(\d+)\]=(\d+)', s, re.M)]
    pairs.sort()
    indices = [v for i, v in pairs] if all(i == j for j, (i, _) in enumerate(pairs)) else []
    return d, indices


def log(message):
    message = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(message, flush=True)
    with (BASE / 'progress.log').open('a') as f:
        f.write(message+'\n')


def record(name, n, m, status, **values):
    row = dict.fromkeys(FIELDS, '')
    row.update(name=name, N=n, M=m, status=status, timestamp=datetime.now().isoformat(timespec='seconds'))
    row.update(values)
    with CSV.open('a', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run(name, n, m, force=False):
    out = BASE / name / f'N{n}_M{m}'
    out.mkdir(parents=True, exist_ok=True)
    output = out / 'run_output.txt'
    stats = out / 'stats.txt'
    indexfile = out / 'indices.txt'
    split = name.startswith('main_')
    if output.exists() and stats.exists() and indexfile.exists():
        d, indices = parse_output(output)
        # Only reuse completed runs, and verify against the peer's saved indices.
        statuses = []
        if CSV.exists():
            with CSV.open(newline='') as f:
                statuses = [r['status'] for r in csv.DictReader(f) if r['name']==name and r['N']==str(n) and r['M']==str(m)]
        if len(indices)==m and statuses and statuses[-1]=='PASS':
            peer = ('main_scalar' if name.startswith('main_rvv_') else
                    'radius_scalar' if name.startswith('radius_rvv_') else
                    'main_scalar' if name == 'ref_rvv_m1' else None)
            peerfile = BASE / peer / f'N{n}_M{m}' / 'indices.txt' if peer else None
            if peerfile is None or not peerfile.exists() or indices == [int(x) for x in peerfile.read_text().splitlines()]:
                log(f'SKIP DONE {name} N={n} M={m}')
                return indices
            log(f'INVALID CACHE {name} N={n} M={m}; peer mismatch')
    if not force:
        hist = history_path(name, n, m)
        if hist is not None and hist.exists():
            # Historical runs are reused only for timing, not assumed to have indices.
            mark = out / 'reused_history.txt'
            if not mark.exists():
                mark.write_text(str(hist.relative_to(ROOT))+'\n')
                try:
                    t, ins = parse_stats(hist, False)
                    record(name, n, m, 'REUSED_HISTORICAL', sim_ms=f'{t:.6f}', sim_insts=ins, source=str(hist.relative_to(ROOT)), note='Single-ROI historical stats; index validation not available here')
                except Exception as e:
                    record(name, n, m, 'HISTORY_PARSE_FAILED', source=str(hist.relative_to(ROOT)), note=str(e))
            log(f'REUSE HISTORY {name} N={n} M={m}')
            return None
    exe = ROOT / 'bin' / BINS[name]
    command = [str(GEM5), f'--outdir={out}', str(CONFIG), str(exe.relative_to(ROOT)), str(DATA.relative_to(ROOT)), str(n), str(m)]
    log(f'RUN {name} N={n} M={m}')
    with output.open('w') as f:
        p = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0 or not stats.exists():
        record(name, n, m, 'FAILED', note=f'gem5 exit={p.returncode}; inspect run_output.txt')
        raise RuntimeError(f'FAIL {name} N={n} M={m}: exit={p.returncode}; {output}')
    d, indices = parse_output(output)
    if len(indices) != m:
        record(name, n, m, 'FAILED_INDEX_OUTPUT', index_count=len(indices), note='Missing or malformed FPS_INDEX lines')
        raise RuntimeError(f'Missing indices {name} N={n} M={m}: {output}')
    indexfile.write_text('\n'.join(map(str, indices))+'\n')
    # Check each new run against the same algorithm's scalar or M1 result
    # whenever that comparison has already completed.
    peer = ('main_scalar' if name.startswith('main_rvv_') else
            'radius_scalar' if name.startswith('radius_rvv_') else
            'main_scalar' if name == 'ref_rvv_m1' else None)
    peer_file = BASE / peer / f'N{n}_M{m}' / 'indices.txt' if peer else None
    if peer_file and peer_file.exists():
        expected = [int(v) for v in peer_file.read_text().splitlines()]
        if indices != expected:
            first = next((i for i,(a,b) in enumerate(zip(expected, indices)) if a != b), None)
            record(name,n,m,'CORRECTNESS_MISMATCH',index_count=len(indices),note=f'vs {peer}; first mismatch step={first}')
            raise RuntimeError(f'CORRECTNESS MISMATCH {name} vs {peer} N={n} M={m} step={first}')
    t, ins = parse_stats(stats, split)
    record(name, n, m, 'PASS', prep_ms=d.get('PREPARATION_MS',''), sample_ms=d.get('SAMPLING_MS',''), total_ms=d.get('TOTAL_MS',''), sim_ms=f'{t:.6f}', sim_insts=ins, index_count=len(indices), source=str(output.relative_to(ROOT)))
    log(f'PASS {name} N={n} M={m} sim={t:.3f}ms')
    return indices


def verify_group(n, m, names, base):
    reference = run(base, n, m)
    for name in names:
        indices = run(name, n, m)
        if indices != reference:
            mismatch = next((i for i, (a, b) in enumerate(zip(reference,indices)) if a!=b), None)
            log(f'CORRECTNESS FAILURE {base} vs {name} N={n} M={m}, first differing step={mismatch}')
            record(name,n,m,'CORRECTNESS_MISMATCH',note=f'vs {base} first differing step={mismatch}')
            raise RuntimeError(f'Correctness mismatch: {name} vs {base}; inspect {BASE}')
        log(f'EXACT MATCH {name} vs {base} N={n} M={m}')


def main():
    missing = [name for name, f in BINS.items() if not (ROOT / 'bin' / f).is_file()]
    if missing:
        sys.exit('MISSING BINARIES: '+', '.join(missing))
    if not DATA.is_file() or not GEM5.is_file() or not CONFIG.is_file():
        sys.exit('Missing KITTI frame, gem5 executable or simulation config')
    log(f'START; {len(CASES)} controlled configurations; {len(LMUL_CASES)} LMUL configurations; resume enabled')
    # Rerunning the preflight reuses previously verified PASS runs.
    verify_group(1024,128,['main_scalar','main_rvv_m1','ref_rvv_m1','main_rvv_m2','main_rvv_m4','main_rvv_m8'], 'ref_scalar')
    verify_group(1024,128,['radius_rvv_m1'], 'radius_scalar')
    log('PREFLIGHT CORRECTNESS PASSED; RadiusFPS LMUL 2/4/8 excluded (incorrect M2 output)')
    # Prioritize newest SphericalFPS scalar/RVV, then controlled baselines.
    for name in ['main_scalar','main_rvv_m1','ref_rvv_m1','radius_rvv_m1','ref_scalar','radius_scalar']:
        for n,m in CASES:
            run(name,n,m)
    # LMUL sensitivity sweep (main and Radius only; no reference LMUL experiments).
    for name in ['main_rvv_m2','main_rvv_m4','main_rvv_m8']:
        for n,m in LMUL_CASES:
            run(name,n,m)
    log('VALIDATED RUNS COMPLETED; RadiusFPS LMUL 2/4/8 were intentionally excluded')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log(f'STOPPED: {exc}')
        raise

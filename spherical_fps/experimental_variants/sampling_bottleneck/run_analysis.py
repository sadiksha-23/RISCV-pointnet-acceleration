#!/usr/bin/env python3
"""Native diagnostic for the unchanged SphericalFPS F2 and its existing profiler.
Run from anywhere inside the repository. Requires a C++17 compiler and KITTI frame.
The instrumented breakdown is diagnostic, not an unbiased performance attribution.
"""
import argparse
import csv
import re
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / 'spherical_fps' / 'main_implementation'
RESULTS = Path(__file__).resolve().parent / 'results'
BIN = ROOT / 'bin'


def compile_program(compiler, name, source):
    output = BIN / name
    command = [compiler, '-O3', '-std=c++17', '-ffp-contract=off', str(SOURCE_DIR / source), '-o', str(output)]
    print('Compiling:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return output


def run_program(exe, dataset, n, m, save_path):
    cmd = [str(exe), str(dataset), str(n), str(m)]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f'Command failed ({completed.returncode}): {cmd}\n{completed.stderr}\n{completed.stdout}')
    save_path.write_text(completed.stdout, encoding='utf-8')
    values = dict(re.findall(r'^([A-Z][A-Z0-9_]+)=(.+)$', completed.stdout, re.M))
    indices = [int(value) for _, value in sorted(((int(i), value) for i, value in re.findall(r'^FPS_INDEX\[(\d+)\]=(\d+)$', completed.stdout, re.M)))]
    if len(indices) != m:
        raise RuntimeError(f'Expected {m} FPS indices in {save_path}; got {len(indices)}')
    return values, indices


def main():
    parser = argparse.ArgumentParser(description='Profile SphericalFPS sampling bottlenecks without changing F2')
    parser.add_argument('--dataset', default='datasets/kitti/0000000000.bin')
    parser.add_argument('--n', type=int, default=100000)
    parser.add_argument('--m', nargs='+', type=int, default=[128, 512])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    if args.n < 1 or args.repeats < 1 or any(m < 1 or m > args.n for m in args.m):
        parser.error('Require N >= 1, repeats >= 1, and 1 <= M <= N')
    if not shutil.which(args.compiler):
        parser.error(f'Compiler not found: {args.compiler}')
    dataset = Path(args.dataset).expanduser()
    dataset = (ROOT / dataset).resolve() if not dataset.is_absolute() else dataset.resolve()
    if not dataset.is_file():
        parser.error(f'KITTI dataset not found: {dataset}')

    RESULTS.mkdir(parents=True, exist_ok=True)
    BIN.mkdir(parents=True, exist_ok=True)
    base = compile_program(args.compiler, 'spherical_fps_bottleneck_baseline', 'spherical_fps.cpp')
    prof = compile_program(args.compiler, 'spherical_fps_bottleneck_profile', 'spherical_fps_profile.cpp')
    columns = ['N', 'M', 'REPEAT', 'BASE_SAMPLING_MS', 'PROFILE_SAMPLING_MS', 'BOUND_MS', 'POINT_UPDATE_MS', 'WINNER_MS', 'RADIUS_MS', 'ACCOUNTED_MS', 'UNACCOUNTED_MS', 'REGION_TESTS', 'REGIONS_SKIPPED', 'POINT_DISTANCE_EVALS', 'INDICES_MATCH']
    rows = []

    for m in args.m:
        for rep in range(1, args.repeats + 1):
            tag = f'N{args.n}_M{m}_run{rep}'
            base_values, base_indices = run_program(base, dataset, args.n, m, RESULTS / f'{tag}_baseline.txt')
            prof_values, prof_indices = run_program(prof, dataset, args.n, m, RESULTS / f'{tag}_profile.txt')
            match = base_indices == prof_indices
            if not match:
                raise RuntimeError(f'FPS output mismatch at N={args.n}, M={m}, repeat={rep}; stopping')
            sample = float(prof_values['SAMPLING_MS'])
            bound = float(prof_values['SAMPLING_BOUND_MS'])
            point = float(prof_values['SAMPLING_POINT_UPDATE_MS'])
            winner = float(prof_values['SAMPLING_WINNER_MS'])
            radius = float(prof_values['SAMPLING_SELECTED_RADIUS_MS'])
            accounted = bound + point + winner + radius
            row = dict(zip(columns, [args.n, m, rep, float(base_values['SAMPLING_MS']), sample, bound, point, winner, radius, accounted, sample - accounted, int(prof_values['REGION_TESTS']), int(prof_values['REGIONS_SKIPPED']), int(prof_values['POINT_DISTANCE_EVALS']), match]))
            rows.append(row)
            print(f'N={args.n} M={m} run={rep}: indices MATCH; baseline sampling={row["BASE_SAMPLING_MS"]:.3f} ms, instrumented sampling={sample:.3f} ms', flush=True)

    with (RESULTS / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print('\nMEDIAN RESULTS (native host only)')
    for m in args.m:
        group = [row for row in rows if row['M'] == m]
        med = lambda name: statistics.median(row[name] for row in group)
        sample = med('PROFILE_SAMPLING_MS')
        print(f'\nM={m}: baseline sampling={med("BASE_SAMPLING_MS"):.3f} ms; profiled sampling={sample:.3f} ms')
        for name in ['BOUND_MS', 'POINT_UPDATE_MS', 'WINNER_MS', 'RADIUS_MS', 'UNACCOUNTED_MS']:
            time_ms = med(name)
            print(f'  {name:<21} {time_ms:10.3f} ms ({100 * time_ms / sample if sample else 0:5.1f}% of instrumented sampling)')
        print(f'  region checks={med("REGION_TESTS"):.0f}; skipped regions={med("REGIONS_SKIPPED"):.0f}; point evaluations={med("POINT_DISTANCE_EVALS"):.0f}')
    print(f'\nRaw output and summary: {RESULTS}')
    print('WARNING: Fine-grained clocks run inside region loops; measured components include substantial clock overhead.')
    print('Do not treat these percentages as exact hardware costs or compare native times directly with gem5.')


if __name__ == '__main__':
    main()

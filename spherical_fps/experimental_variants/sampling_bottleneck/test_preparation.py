#!/usr/bin/env python3
"""Diagnose stages of F2 spherical preparation without changing the algorithm.

Place at spherical_fps/experimental_variants/preparation_profile/test_preparation.py
Run in VS Code: python spherical_fps/experimental_variants/preparation_profile/test_preparation.py

Native host measurements only. Original F2 is never modified. Creates a generated
instrumented C++ source and compiled binaries in the local results/ folder.
"""
from __future__ import annotations

import csv
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'spherical_fps/main_implementation/spherical_fps.cpp'
DATA = ROOT / 'datasets/kitti/0000000000.bin'
OUT = Path(__file__).resolve().parent / 'results'
COMPILER = 'g++'
FLAGS = ['-O3', '-std=c++17', '-ffp-contract=off']
N = 100000
M_VALUES = [128, 512]
REPEATS = 7

# These markers match the committed F2 source. Abort if source differs; never
# silently instrument the wrong function or version.

def replace_once(text: str, before: str, after: str) -> str:
    count = text.count(before)
    if count != 1:
        raise RuntimeError(f'Expected exactly one source marker (found {count}): {before[:100]!r}')
    return text.replace(before, after, 1)


def make_profile_source(source: str) -> str:
    declarations = r'''
struct PreparationStages {
    double setup_ms=0, classify_ms=0, radial_edges_ms=0, assign_ms=0, finalize_ms=0;
};
static PreparationStages prep_stages;
static inline double profile_elapsed(std::chrono::steady_clock::time_point a,
                                     std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
'''
    source = replace_once(source, 'static std::vector<Region> build_regions(', declarations + '\nstatic std::vector<Region> build_regions(')

    source = replace_once(source,
        '    const size_t bins=static_cast<size_t>(AZ_BINS)*EL_BINS*RAD_BINS;\n',
        '    const auto stage_begin=std::chrono::steady_clock::now();\n'
        '    const size_t bins=static_cast<size_t>(AZ_BINS)*EL_BINS*RAD_BINS;\n')

    source = replace_once(source,
        '    for(int i=0;i<n;++i){const size_t p=static_cast<size_t>(i)*3;\n',
        '    const auto classify_begin=std::chrono::steady_clock::now();\n'
        '    prep_stages.setup_ms=profile_elapsed(stage_begin,classify_begin);\n'
        '    for(int i=0;i<n;++i){const size_t p=static_cast<size_t>(i)*3;\n')

    source = replace_once(source,
        '    // Precompute squared radial boundaries once, not one square root per point.\n',
        '    const auto classify_end=std::chrono::steady_clock::now();\n'
        '    prep_stages.classify_ms=profile_elapsed(classify_begin,classify_end);\n'
        '    // Precompute squared radial boundaries once, not one square root per point.\n')

    source = replace_once(source,
        '    for (int i=0; i<n; ++i) {\n',
        '    const auto assign_begin=std::chrono::steady_clock::now();\n'
        '    prep_stages.radial_edges_ms=profile_elapsed(classify_end,assign_begin);\n'
        '    for (int i=0; i<n; ++i) {\n')

    source = replace_once(source,
        '    std::vector<Region> occupied;occupied.reserve(std::min(bins,static_cast<size_t>(n)));\n',
        '    const auto assign_end=std::chrono::steady_clock::now();\n'
        '    prep_stages.assign_ms=profile_elapsed(assign_begin,assign_end);\n'
        '    std::vector<Region> occupied;occupied.reserve(std::min(bins,static_cast<size_t>(n)));\n')

    source = replace_once(source,
        '    return occupied;\n}\nstatic double lower_bound(',
        '    const auto finalize_end=std::chrono::steady_clock::now();\n'
        '    prep_stages.finalize_ms=profile_elapsed(assign_end,finalize_end);\n'
        '    return occupied;\n}\nstatic double lower_bound(')

    source = replace_once(source,
        '        double origin[3];calculate_centroid(xyz,n,origin);\n',
        '        double origin[3];calculate_centroid(xyz,n,origin);\n'
        '        const auto centroid_end=std::chrono::steady_clock::now();\n')

    source = replace_once(source,
        '        std::printf("PREPARATION_MS=%.6f\\nSAMPLING_MS=%.6f\\nTOTAL_MS=%.6f\\n",prep_ms,sample_ms,prep_ms+sample_ms);\n',
        '        std::printf("PREPARATION_MS=%.6f\\nSAMPLING_MS=%.6f\\nTOTAL_MS=%.6f\\n",prep_ms,sample_ms,prep_ms+sample_ms);\n'
        '        std::printf("PREP_CENTROID_MS=%.6f\\nPREP_SETUP_MS=%.6f\\nPREP_CLASSIFY_MS=%.6f\\nPREP_RADIAL_EDGES_MS=%.6f\\nPREP_ASSIGN_MS=%.6f\\nPREP_FINALIZE_MS=%.6f\\n",\n'
        '            profile_elapsed(start_prep,centroid_end),prep_stages.setup_ms,prep_stages.classify_ms,prep_stages.radial_edges_ms,prep_stages.assign_ms,prep_stages.finalize_ms);\n')
    return source


def run(args: list[str]) -> str:
    finished = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, errors='replace')
    if finished.returncode:
        raise RuntimeError('Command failed: ' + ' '.join(args) + '\n' + finished.stdout + '\n' + finished.stderr)
    return finished.stdout


def fields(output: str) -> dict[str, str]:
    return dict(re.findall(r'^([A-Z_]+(?:\[\d+\])?)=(.+)$', output, flags=re.MULTILINE))


def fps_indices(output: str) -> list[int]:
    pairs = [(int(i), int(v)) for i, v in re.findall(r'^FPS_INDEX\[(\d+)\]=(\d+)$', output, re.MULTILINE)]
    pairs.sort()
    if not pairs or any(i != k for k, (i, _) in enumerate(pairs)):
        raise RuntimeError('Missing or discontinuous FPS_INDEX lines')
    return [v for _, v in pairs]


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit('Missing F2 source: ' + str(SOURCE))
    if not DATA.is_file():
        raise SystemExit('Missing KITTI frame: ' + str(DATA))
    OUT.mkdir(parents=True, exist_ok=True)

    diagnostic_cpp = OUT / 'spherical_fps_preparation_profile.cpp'
    diagnostic_cpp.write_text(make_profile_source(SOURCE.read_text(encoding='utf-8')), encoding='utf-8')
    original_exe = OUT / 'f2_original.exe'
    profile_exe = OUT / 'f2_preparation_profile.exe'
    for src, exe in [(SOURCE, original_exe), (diagnostic_cpp, profile_exe)]:
        cmd = [COMPILER, *FLAGS, str(src), '-o', str(exe)]
        print('Compiling:', ' '.join(cmd), flush=True)
        run(cmd)

    keys = ['PREP_CENTROID_MS','PREP_SETUP_MS','PREP_CLASSIFY_MS',
            'PREP_RADIAL_EDGES_MS','PREP_ASSIGN_MS','PREP_FINALIZE_MS']
    rows: list[dict[str, object]] = []
    for m in M_VALUES:
        for trial in range(1, REPEATS + 1):
            # Alternate the order to limit ordering bias from native host conditions.
            cmds = [('original', original_exe), ('profile', profile_exe)]
            if trial % 2 == 0:
                cmds.reverse()
            outputs = {name: run([str(exe), str(DATA), str(N), str(m)]) for name, exe in cmds}
            if fps_indices(outputs['original']) != fps_indices(outputs['profile']):
                raise RuntimeError(f'FPS indices differ at M={m}, run={trial}. Original F2 left untouched.')
            a, b = fields(outputs['original']), fields(outputs['profile'])
            for counter in ['REGION_TESTS', 'REGIONS_SKIPPED', 'POINT_DISTANCE_EVALS']:
                if a[counter] != b[counter]:
                    raise RuntimeError(f'Counter {counter} differs at M={m}, trial={trial}')
            row: dict[str, object] = {'N': N, 'M': m, 'run': trial,
                'original_prep_ms': float(a['PREPARATION_MS']),
                'original_sampling_ms': float(a['SAMPLING_MS']),
                'profile_prep_ms': float(b['PREPARATION_MS']),
                'profile_sampling_ms': float(b['SAMPLING_MS'])}
            row.update({key.lower(): float(b[key]) for key in keys})
            rows.append(row)
            print(f'N={N} M={m} run={trial}: indices MATCH; original prep={row["original_prep_ms"]:.3f} ms; profiled prep={row["profile_prep_ms"]:.3f} ms', flush=True)

        subset = [r for r in rows if r['M'] == m]
        median = lambda key: statistics.median(float(r[key]) for r in subset)
        profile_prep = median('profile_prep_ms')
        original_prep = median('original_prep_ms')
        print(f'\nM={m} MEDIAN native preparation: original={original_prep:.3f} ms, instrumented={profile_prep:.3f} ms')
        stage_sum = 0.0
        for key in keys:
            value = median(key.lower())
            stage_sum += value
            print(f'  {key:<25} {value:8.3f} ms ({100 * value / profile_prep:5.1f}% of instrumented prep)')
        print(f'  Approximate sum of stage medians: {stage_sum:.3f} ms\n')

    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved generated C++, binaries, and summary.csv to:', OUT)
    print('NOTE: Native measurements only, not gem5. The per-stage percentages are approximate.')
    print('NOTE: Preparation should be essentially independent of M; differences indicate native timing noise.')
    print('Original F2 source was not modified.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare exact F2 elevation classifiers; run from any working directory on Windows/WSL.

This experiment never modifies the original F2 implementation. Generated C++ and
executables live in results/elevation_variants/ next to this runner.
"""
from __future__ import annotations

import csv
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def find_root(script: Path) -> Path:
    for parent in [script.parent, *script.parents]:
        if (parent / 'spherical_fps/main_implementation/spherical_fps.cpp').is_file():
            return parent
    raise SystemExit('ERROR: Could not find repository root containing F2 source.')


ROOT = find_root(Path(__file__).resolve())
SOURCE = ROOT / 'spherical_fps/main_implementation/spherical_fps.cpp'
DATA = ROOT / 'datasets/kitti/0000000000.bin'
OUT = Path(__file__).resolve().parent / 'results/elevation_variants'
OUT.mkdir(parents=True, exist_ok=True)
COMPILER = shutil.which('g++')
if not COMPILER:
    raise SystemExit('ERROR: g++ is not on PATH. Use your VS Code terminal with a C++ compiler.')
if not DATA.is_file():
    raise SystemExit(f'ERROR: KITTI frame missing: {DATA}')

# Use the exact same operations as F2's original squared-comparison predicate.
# The only changes are the ordering/branching of the elevation boundary tests.
COMMON = r'''static inline bool f2_elevation_above(double horizontal_sq, double z, int mid) {
    static const std::vector<std::pair<double,double>> boundaries=[] {
        std::vector<std::pair<double,double>> edges;
        for(int k=0;k<=EL_BINS;++k) {
            const double theta=-PI/2.0+(PI*k)/EL_BINS;
            edges.emplace_back(std::sin(theta),std::cos(theta));
        }
        return edges;
    }();
    const auto& edge=boundaries[mid];
    const double lhs=z*edge.second;
    const double sin_theta=edge.first;
    bool above;
    if (sin_theta == 0.0) above=lhs >= 0.0;
    else if (sin_theta > 0.0) {
        above=lhs >= 0.0 && lhs*lhs >= horizontal_sq*sin_theta*sin_theta;
    } else {
        above=lhs >= 0.0 || lhs*lhs <= horizontal_sq*sin_theta*sin_theta;
    }
    return above;
}
'''

# The original function is retained as-is in the reference build. These two
# replacements intentionally specialize to exactly 8 elevation bins.
SIGN_FIRST = COMMON + r'''
static int direct_elevation_bin(double x,double y,double z) {
    const double horizontal_sq=x*x+y*y;
    if(horizontal_sq==0.0) return z>=0 ? EL_BINS-1 : 0;
    // The middle boundary is exactly 0 degrees: equivalent to z>=0 here.
    if(z>=0.0) {
        if(f2_elevation_above(horizontal_sq,z,6))
            return f2_elevation_above(horizontal_sq,z,7) ? 7 : 6;
        return f2_elevation_above(horizontal_sq,z,5) ? 5 : 4;
    }
    if(f2_elevation_above(horizontal_sq,z,2))
        return f2_elevation_above(horizontal_sq,z,3) ? 3 : 2;
    return f2_elevation_above(horizontal_sq,z,1) ? 1 : 0;
}
'''

FIXED_TREE = COMMON + r'''
static int direct_elevation_bin(double x,double y,double z) {
    const double horizontal_sq=x*x+y*y;
    if(horizontal_sq==0.0) return z>=0 ? EL_BINS-1 : 0;
    // Fully unrolled decision tree, same F2 comparison at every boundary.
    if(f2_elevation_above(horizontal_sq,z,4)) {
        if(f2_elevation_above(horizontal_sq,z,6)) {
            if(f2_elevation_above(horizontal_sq,z,7)) return 7;
            return 6;
        }
        if(f2_elevation_above(horizontal_sq,z,5)) return 5;
        return 4;
    }
    if(f2_elevation_above(horizontal_sq,z,2)) {
        if(f2_elevation_above(horizontal_sq,z,3)) return 3;
        return 2;
    }
    if(f2_elevation_above(horizontal_sq,z,1)) return 1;
    return 0;
}
'''


def run(cmd: list[str], capture: bool = False) -> str:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture, check=False)
    if result.returncode:
        if capture:
            print(result.stdout[-2000:])
            print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f'Command failed ({result.returncode}): {cmd}')
    return result.stdout if capture else ''


def replace_elevation(source: str, replacement: str) -> str:
    match = re.search(r'static int direct_elevation_bin\s*\(double x,double y,double z\)\s*\{', source)
    if not match:
        raise RuntimeError('Cannot locate direct_elevation_bin() in F2 source.')
    opening = source.index('{', match.start())
    depth = 0
    end = None
    for pos in range(opening, len(source)):
        if source[pos] == '{':
            depth += 1
        elif source[pos] == '}':
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
    if end is None:
        raise RuntimeError('Could not find end of original elevation function.')
    return source[:match.start()] + replacement + source[end:]


def extract(text: str) -> dict[str, object]:
    def val(key: str) -> float:
        m = re.search(r'^' + re.escape(key) + r'=([^\r\n]+)', text, re.M)
        if not m:
            raise RuntimeError(f'Missing {key} in program output')
        return float(m.group(1))
    indices = [int(v) for v in re.findall(r'^FPS_INDEX\[\d+\]=(-?\d+)', text, re.M)]
    return {
        'prep': val('PREPARATION_MS'),
        'sample': val('SAMPLING_MS'),
        'total': val('TOTAL_MS'),
        'indices': indices,
        'checks': int(val('REGION_TESTS')),
        'skipped': int(val('REGIONS_SKIPPED')),
        'evals': int(val('POINT_DISTANCE_EVALS')),
    }


if __name__ == '__main__':
    source = SOURCE.read_text(encoding='utf-8')
    # We are using a specialized 8-bin tree; do not silently run for other setups.
    if re.search(r'#define EL_BINS\s+(?!8\b)\d+', source):
        raise SystemExit('ERROR: This experiment requires EL_BINS=8.')
    if not re.search(r'#define EL_BINS\s+8\b', source):
        raise SystemExit('ERROR: Cannot verify EL_BINS=8 in the F2 source.')
    variants = {'f2': source, 'sign_first': replace_elevation(source, SIGN_FIRST), 'fixed_tree': replace_elevation(source, FIXED_TREE)}
    binaries = {}
    for name, content in variants.items():
        cpp = OUT / f'f2_elevation_{name}.cpp'
        exe = OUT / (f'f2_elevation_{name}.exe' if sys.platform == 'win32' else f'f2_elevation_{name}')
        cpp.write_text(content, encoding='utf-8')
        print('Compiling', name, flush=True)
        run([COMPILER, '-O3', '-std=c++17', '-ffp-contract=off', str(cpp), '-o', str(exe)])
        binaries[name] = exe

    # Strong classifier-level test: original F2 versus both variants, including
    # axes and floats directly adjacent to all elevation boundaries.
    # The same source is compiled into this standalone test program, so checks
    # use the actual F2 and generated classifier code, not Python approximations.
    prefix = source[:source.index('struct Region {')]
    m = re.search(r'static int direct_elevation_bin\s*\(double x,double y,double z\)\s*\{', prefix)
    if not m:
        raise RuntimeError('Cannot locate F2 classifier in header source.')
    # Rename original function and reuse all its F2 helper types, constants.
    test_cpp = re.sub(r'static int direct_elevation_bin\(', 'static int elevation_reference(', prefix, count=1)
    sign_code = SIGN_FIRST.replace('f2_elevation_above', 'sign_elevation_above').replace('direct_elevation_bin(', 'elevation_sign_first(')
    fixed_code = FIXED_TREE.replace('f2_elevation_above', 'tree_elevation_above').replace('direct_elevation_bin(', 'elevation_fixed_tree(')
    test_cpp += '\n' + sign_code + '\n' + fixed_code + '\n'
    test_cpp += r'''
#include <random>
int main() {
    uint64_t count=0, mismatches=0;
    auto check=[&](double x,double y,double z) {
        int a=elevation_reference(x,y,z), b=elevation_sign_first(x,y,z), c=elevation_fixed_tree(x,y,z);
        ++count;
        if(a!=b || a!=c) {
            if(mismatches<8) std::printf("MISMATCH x=%.17g y=%.17g z=%.17g ref=%d sign=%d tree=%d\n",x,y,z,a,b,c);
            ++mismatches;
        }
    };
    check(0,0,0);check(0,0,1);check(0,0,-1);
    check(1,0,0);check(-1,0,0);check(0,1,0);check(0,-1,0);
    for(int k=0;k<=8;++k) {
        double a=-PI/2.0+(PI*k)/8.0;
        for(double az : {0.0,0.1,0.6,1.4,3.1,4.8}) {
            double x=std::cos(az)*std::cos(a),y=std::sin(az)*std::cos(a),z=std::sin(a);
            check(x,y,z);
            check(x,y,std::nextafter(z,-std::numeric_limits<double>::infinity()));
            check(x,y,std::nextafter(z,std::numeric_limits<double>::infinity()));
            check(x*1000,y*1000,z*1000);
        }
    }
    std::mt19937_64 rng(20261121);
    std::uniform_real_distribution<double> dist(-1000,1000);
    for(int i=0;i<300000;++i) check(dist(rng),dist(rng),dist(rng));
    std::printf("CLASSIFIER_CASES=%llu\nCLASSIFIER_MISMATCHES=%llu\n",(unsigned long long)count,(unsigned long long)mismatches);
    return mismatches ? 1 : 0;
}
'''
    test_path = OUT / 'elevation_classifier_check.cpp'
    test_bin = OUT / ('elevation_classifier_check.exe' if sys.platform == 'win32' else 'elevation_classifier_check')
    test_path.write_text(test_cpp, encoding='utf-8')
    print('Compiling classifier correctness check', flush=True)
    run([COMPILER, '-O3', '-std=c++17', '-ffp-contract=off', str(test_path), '-o', str(test_bin)])
    print(run([str(test_bin)], capture=True).strip(), flush=True)

    records = []
    rng = random.Random(20261121)
    for m_count in (128, 256, 512, 1024):
        for repeat in range(1, 8):
            names = list(binaries)
            rng.shuffle(names)  # Avoid always penalizing the last implementation.
            results = {}
            for name in names:
                output = run([str(binaries[name]), str(DATA), '100000', str(m_count)], capture=True)
                (OUT / f'N100000_M{m_count}_run{repeat}_{name}.txt').write_text(output, encoding='utf-8')
                results[name] = extract(output)
            ref = results['f2']
            for name in ('sign_first', 'fixed_tree'):
                item = results[name]
                if item['indices'] != ref['indices']:
                    raise RuntimeError(f'FPS indices DIFFER: {name} M={m_count} repeat={repeat}')
                if (item['checks'], item['skipped'], item['evals']) != (ref['checks'], ref['skipped'], ref['evals']):
                    raise RuntimeError(f'FPS work counters DIFFER: {name} M={m_count} repeat={repeat}')
            for name, item in results.items():
                records.append({'N': 100000, 'M': m_count, 'repeat': repeat, 'variant': name,
                                'prep_ms': item['prep'], 'sampling_ms': item['sample'], 'total_ms': item['total']})
            print(f'M={m_count} repeat={repeat}: indices and work counters MATCH; ' +
                  ', '.join(f'{k}={results[k]["total"]:.3f} ms' for k in ('f2', 'sign_first', 'fixed_tree')), flush=True)
        print(f'\nM={m_count} MEDIANS (native ms):')
        med = {}
        for name in ('f2', 'sign_first', 'fixed_tree'):
            selected = [r for r in records if r['M'] == m_count and r['variant'] == name]
            med[name] = {metric: statistics.median(r[metric] for r in selected) for metric in ('prep_ms','sampling_ms','total_ms')}
        for name in ('f2', 'sign_first', 'fixed_tree'):
            d = med[name]
            reduction = 100 * (1-d['total_ms']/med['f2']['total_ms'])
            print(f'  {name:11s} prep={d["prep_ms"]:.3f}, sample={d["sampling_ms"]:.3f}, total={d["total_ms"]:.3f}, reduction={reduction:+.1f}%')
        print()

    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['N','M','repeat','variant','prep_ms','sampling_ms','total_ms'])
        writer.writeheader()
        writer.writerows(records)
    print('All classifier cases and FPS runs MATCH.')
    print('Results:', OUT)
    print('NOTE: Native Windows measurements can be noisy; validate promising variants on gem5.')
    print('Original F2 was NOT modified.')

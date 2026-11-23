#!/usr/bin/env python3
"""Diagnostic for F2 SphericalFPS bound effectiveness; does not modify the original source.

Run from VS Code or any working directory:
    python spherical_fps/experimental_variants/sampling_bottleneck/test_bound_contributions.py

Requires a C++17 g++ compiler and datasets/kitti/0000000000.bin.
Times of diagnostic builds are NOT comparable to the uninstrumented baseline.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "spherical_fps/ref_implementation/spherical_fps.cpp"
DEFAULT_DATA = ROOT / "datasets/kitti/0000000000.bin"
OUTDIR = Path(__file__).resolve().parent / "results" / "bound_contributions"

DIAGNOSTICS = r'''
// DIAGNOSTIC ONLY: Each bound is assessed independently against the original
// max-distance threshold. These counters never affect skipping or selection.
struct BoundDiagnostics {
    uint64_t checks=0, original_skips=0;
    uint64_t aabb=0, sphere=0, radial=0;
    uint64_t aabb_or_sphere=0, aabb_or_radial=0;
    uint64_t any_single=0;
    uint64_t sphere_adds_after_aabb=0, radial_adds_after_aabb_sphere=0;
    uint64_t aabb_adds_after_sphere_radial=0;
    uint64_t only_aabb=0, only_sphere=0, only_radial=0;
    uint64_t none=0;
} bound_diag;

static void record_bound_diagnostics(const Region& g, const std::vector<float>& xyz,
                                     int newest, double selected_radius,
                                     double max_sq, double margin, bool original_skip) {
    const size_t p=static_cast<size_t>(newest)*3;
    const double x=static_cast<double>(xyz[p]);
    const double y=static_cast<double>(xyz[p+1]);
    const double z=static_cast<double>(xyz[p+2]);
    const double dx=x-g.cx,dy=y-g.cy,dz=z-g.cz;
    const double center_dist=std::sqrt(dx*dx+dy*dy+dz*dz);
    const double sphere_lb=std::max(0.0,center_dist-g.radius);
    const double radial_lb=std::max({0.0,g.radial_min-selected_radius,selected_radius-g.radial_max});
    const double bx=std::max({g.xmin-x,0.0,x-g.xmax});
    const double by=std::max({g.ymin-y,0.0,y-g.ymax});
    const double bz=std::max({g.zmin-z,0.0,z-g.zmax});
    const double aabb_lb=std::sqrt(bx*bx+by*by+bz*bz);

    // Match the scalar F2's conservative adjustment for each individual bound.
    // Do NOT use these diagnostic booleans to control the original algorithm.
    const auto qualifies=[&](double raw) {
        const double adjusted=std::max(0.0,raw-1e-6*(1.0+raw));
        return adjusted*adjusted > max_sq+margin;
    };
    const bool a=qualifies(aabb_lb);
    const bool s=qualifies(sphere_lb);
    const bool r=qualifies(radial_lb);
    ++bound_diag.checks;
    bound_diag.original_skips+=original_skip;
    bound_diag.aabb+=a;
    bound_diag.sphere+=s;
    bound_diag.radial+=r;
    bound_diag.aabb_or_sphere+=(a||s);
    bound_diag.aabb_or_radial+=(a||r);
    bound_diag.any_single+=(a||s||r);
    bound_diag.sphere_adds_after_aabb+=(!a&&s);
    bound_diag.radial_adds_after_aabb_sphere+=(!a&&!s&&r);
    bound_diag.aabb_adds_after_sphere_radial+=(!s&&!r&&a);
    bound_diag.only_aabb+=(a&&!s&&!r);
    bound_diag.only_sphere+=(!a&&s&&!r);
    bound_diag.only_radial+=(!a&&!s&&r);
    bound_diag.none+=(!a&&!s&&!r);
}

static void print_bound_diagnostics() {
    const double n=bound_diag.checks ? static_cast<double>(bound_diag.checks) : 1.0;
    const auto report=[&](const char* key, uint64_t count) {
        std::printf("BOUND_%s=%llu (%.3f%% of region checks)\\n",key,
                    (unsigned long long)count,100.0*static_cast<double>(count)/n);
    };
    report("REGION_CHECKS",bound_diag.checks);
    report("ORIGINAL_SKIPS",bound_diag.original_skips);
    report("AABB_ALONE",bound_diag.aabb);
    report("SPHERE_ALONE",bound_diag.sphere);
    report("RADIAL_ALONE",bound_diag.radial);
    report("AABB_OR_SPHERE",bound_diag.aabb_or_sphere);
    report("AABB_OR_RADIAL",bound_diag.aabb_or_radial);
    report("ANY_SINGLE",bound_diag.any_single);
    report("SPHERE_EXTRA_AFTER_AABB",bound_diag.sphere_adds_after_aabb);
    report("RADIAL_EXTRA_AFTER_AABB_SPHERE",bound_diag.radial_adds_after_aabb_sphere);
    report("AABB_EXTRA_AFTER_SPHERE_RADIAL",bound_diag.aabb_adds_after_sphere_radial);
    report("ONLY_AABB",bound_diag.only_aabb);
    report("ONLY_SPHERE",bound_diag.only_sphere);
    report("ONLY_RADIAL",bound_diag.only_radial);
    report("NO_BOUND_PASSES",bound_diag.none);
    std::printf("BOUND_MISMATCH=%llu\\n",
      (unsigned long long)(bound_diag.original_skips>bound_diag.any_single
        ? bound_diag.original_skips-bound_diag.any_single
        : bound_diag.any_single-bound_diag.original_skips));
}
'''
# DIAGNOSTICS is a raw string to preserve C++ syntax; turn escaped newlines into C++ escapes.
DIAGNOSTICS = DIAGNOSTICS.replace('\\\\n"', '\\n"')

INSERT_BEFORE = "static std::vector<int> spherical_fps("
NEEDLE = "if (lb*lb > max_sq+margin) {"
AFTER = 'std::printf("CORRECTNESS=NOT_CHECKED (performance-only run)\\n");return 0;'


def make_source(source: str) -> str:
    for name, needle in [("sampling function", INSERT_BEFORE), ("region decision", NEEDLE), ("final output", AFTER)]:
        if source.count(needle) != 1:
            raise RuntimeError(f"Source changed: expected one {name}, found {source.count(needle)}. Stop rather than instrumenting the wrong algorithm.")
    source = source.replace(INSERT_BEFORE, DIAGNOSTICS + "\n" + INSERT_BEFORE, 1)
    source = source.replace(NEEDLE,
        "record_bound_diagnostics(g,xyz,newest,selected_radius,max_sq,margin,lb*lb > max_sq+margin);\n            " + NEEDLE, 1)
    source = source.replace(AFTER, "print_bound_diagnostics();\n        " + AFTER, 1)
    return source


def run(cmd):
    print("Running:", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(x) for x in cmd], check=True, text=True, capture_output=True)


def indices(output: str):
    return [int(v) for _, v in re.findall(r"^FPS_INDEX\[(\d+)\]=(\d+)$", output, re.M)]


def metric(output: str, name: str):
    match = re.search(r"^BOUND_" + re.escape(name) + r"=(\d+)", output, re.M)
    if not match:
        raise RuntimeError(f"Missing BOUND_{name} in diagnostic output")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="Measure the independent contribution of F2's existing bounds")
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--m", type=int, nargs="+", default=[128,512])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if not SOURCE.exists() or not args.data.exists():
        parser.error(f"Missing source or dataset: {SOURCE} ; {args.data}")
    if args.n < 1 or any(m < 1 or m > args.n for m in args.m):
        parser.error("Require 1 <= M <= N")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    generated = OUTDIR / "f2_bound_diagnostic.cpp"
    generated.write_text(make_source(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    extension = ".exe" if os.name == "nt" else ""
    base = OUTDIR / ("f2_original" + extension)
    diag = OUTDIR / ("f2_bound_diagnostic" + extension)
    flags = ["g++", "-O3", "-std=c++17", "-ffp-contract=off"]
    for src, executable in [(SOURCE,base),(generated,diag)]:
        run([*flags, src,"-o",executable])
    print("\nDIAGNOSTIC COUNTS (native CPU; not performance timings)")
    for m in args.m:
        original=run([base,args.data,args.n,m]).stdout
        analyzed=run([diag,args.data,args.n,m]).stdout
        (OUTDIR / f"N{args.n}_M{m}_baseline.txt").write_text(original,encoding="utf-8")
        (OUTDIR / f"N{args.n}_M{m}_diagnostic.txt").write_text(analyzed,encoding="utf-8")
        matching=indices(original)==indices(analyzed) and len(indices(original))==m
        if not matching:
            raise RuntimeError(f"M={m}: FPS indices DIFFER! Inspect saved outputs.")
        checks=metric(analyzed,"REGION_CHECKS")
        skipped=metric(analyzed,"ORIGINAL_SKIPS")
        a=metric(analyzed,"AABB_ALONE")
        s=metric(analyzed,"SPHERE_ALONE")
        r=metric(analyzed,"RADIAL_ALONE")
        extra_s=metric(analyzed,"SPHERE_EXTRA_AFTER_AABB")
        extra_r=metric(analyzed,"RADIAL_EXTRA_AFTER_AABB_SPHERE")
        combined=metric(analyzed,"ANY_SINGLE")
        mismatch=metric(analyzed,"MISMATCH")
        print(f"\nN={args.n} M={m}: indices MATCH; checks={checks:,}; original skips={skipped:,}")
        print(f"  AABB alone: {a:,} ({100*a/max(1,skipped):.1f}% of skips)")
        print(f"  Sphere alone: {s:,} ({100*s/max(1,skipped):.1f}% of skips)")
        print(f"  Radial alone: {r:,} ({100*r/max(1,skipped):.1f}% of skips)")
        print(f"  Extra skips from sphere after AABB: {extra_s:,}")
        print(f"  Extra skips from radial after AABB+sphere: {extra_r:,}")
        print(f"  All three combined: {combined:,}; classification mismatch={mismatch}")
        if mismatch:
            print("  WARNING: numerical edge case; inspect diagnostic output before interpreting counts")
    print(f"\nSaved generated C++ and full outputs to: {OUTDIR}")
    print("The original F2 implementation was NOT changed. Diagnostic runtimes must NOT be used as speedup estimates.")

if __name__=="__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        else:
            print("ERROR:",exc,file=sys.stderr)
        sys.exit(1)

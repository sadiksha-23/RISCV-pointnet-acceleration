#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
python3 spherical_fps/experimental_variants/optimization_tests/run_prep_tests.py --mode native "$@"

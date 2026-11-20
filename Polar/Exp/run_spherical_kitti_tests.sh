#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 Polar/experiment/spherical_kitti_tests.py "$@"

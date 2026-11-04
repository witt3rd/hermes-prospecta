#!/usr/bin/env bash
# Tier 1 (gates landing; target < 60 s): the lander's own self-test, then the whole pytest suite in 8 parallel workers (Postgres via testcontainers, one container per worker).
# Tier 2 (never gates landing) is the same suite run serially, on a schedule: see .github/workflows/ci.yml.
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/test_land_direct.sh
uv run --extra test --with pytest-xdist pytest -q -n 8

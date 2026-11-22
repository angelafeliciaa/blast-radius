#!/bin/sh
# Runs sqlglot's suite four times under tools/mapgen_plugin.py and copies the raw
# pickles to $1: every test in its own fork (no cross-test caches), all tests in
# one process (order effects), forked with the comparison profile on, and forked
# with key-level recording of class-level dicts, and every test module imported in its
# own fork (what collection touches, per module), once with sqlglot fully imported and
# once with only `import sqlglot` done; then the tests again, one fork each, without the
# eager import of every module (lazy loading paths), and every test alone in a fork of a
# process that has only imported sqlglot (no caches from collection or earlier tests).
# The missing_* runs record, per test, the arguments that were None or empty when an
# expression was validated (what a flip of required-ness in arg_types can reach).
# ONLY=<label> runs just one.
# Needs the dev image (tag: $IMAGE, built from tools/Dockerfile.dev). CPUS sets the
# container CPUs and the number of forked test runners (default 8).
set -eu
OUT=${1:?usage: gen_raw_maps.sh <outdir>}
IMAGE=${IMAGE:-br-dev-arm}
PLATFORM=${PLATFORM:-linux/arm64}
TOOLS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
APP=$(cd "$TOOLS/../app" && pwd)
run() {
  label=$1; mode=$2; profile=$3; keys=${4:-0}; missing=${5:-0}
  name="mapgen-$label-$$"
  docker run --name "$name" --platform "$PLATFORM" --cpus "${CPUS:-8}" \
    -v "$TOOLS":/tools:ro -v "$APP":/app:ro -v "$TOOLS/../sample/tests.json":/tests.json:ro -e MAPGEN_TESTS=/tests.json -e PYTHONPATH=/opt/sqlglot:/tools -e SKIP_INTEGRATION=0 \
    -e MAPGEN_OUT=/tmp/map.pkl -e MAPGEN_ROOT=/opt/sqlglot -e MAPGEN_MODE=$mode \
    -e MAPGEN_PROFILE=$profile -e MAPGEN_PROFILE_NODES="${PROFILE_NODES:-0}" -e MAPGEN_KEYS=$keys -e MAPGEN_MISSING=$missing -e MAPGEN_APP=/app -e MAPGEN_JOBS="${CPUS:-8}" \
    -e MAPGEN_DEFAULTS="${DEFAULTS:-0}" \
    "$IMAGE" sh -c 'cd /opt/sqlglot && python -m pytest -q -p no:cacheprovider -p mapgen_plugin tests > /tmp/log.txt 2>&1; tail -3 /tmp/log.txt'
  docker cp "$name:/tmp/map.pkl" "$OUT/map_$label.pkl"
  docker rm "$name" > /dev/null
}
if [ -n "${ONLY:-}" ]; then
  case "$ONLY" in
    fork) run fork fork 0 ;; seq) run seq seq 0 ;; profile) run profile fork 1 ;; keys) run keys fork 0 1 ;;
    collect) run collect collect 0 1 ;; collect_lazy) run collect_lazy collect_lazy 0 0 ;;
    fork_lazy) run fork_lazy fork_lazy 0 0 ;; fresh) run fresh fresh 0 0 ;;
    missing) run missing fork 0 0 1 ;; missing_seq) run missing_seq seq 0 0 1 ;;
    missing_collect) run missing_collect collect 0 0 1 ;; missing_fresh) run missing_fresh fresh 0 0 1 ;;
    items) run items items 0 0 ;;
    defaults) DEFAULTS=1; run defaults fork 0 ;; defaults_seq) DEFAULTS=1; run defaults_seq seq 0 ;;
  esac
else
  run fork fork 0
  run seq seq 0
  run profile fork 1
  run keys fork 0 1
  run collect collect 0 1
  run collect_lazy collect_lazy 0 0
  run fork_lazy fork_lazy 0 0
  run fresh fresh 0 0
  run missing fork 0 0 1
  run missing_seq seq 0 0 1
  run missing_collect collect 0 0 1
  run missing_fresh fresh 0 0 1
  run items items 0 0
fi
ls -la "$OUT"

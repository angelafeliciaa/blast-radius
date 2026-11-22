#!/bin/sh
# Runs the image once on the sample under the scorer's limits, like `make sample`,
# but also reads the container's cgroup CPU usage so the total includes CPU time.
set -eu
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-blast-radius-entry}
OUT=${OUT:-output.jsonl}
ERR=${ERR:-.work/run.err}
mkdir -p .work
start=$(date +%s)
docker run --rm --platform linux/amd64 --network none --cpus 2 \
  --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp ${EXTRA_ENV:-} \
  --volume "$PWD/sample/commits:/data/commits:ro" \
  --volume "$PWD/sample/tests.json:/data/tests.json:ro" \
  "$IMAGE" sh -c 'python /app/selector.py; status=$?; echo "CPUSTAT $(grep usage_usec /sys/fs/cgroup/cpu.stat)" >&2; exit $status' \
  > "$OUT" 2> "$ERR"
end=$(date +%s)
cpu_usec=$(grep '^CPUSTAT' "$ERR" | awk '{print $3}')
if [ -z "$cpu_usec" ]; then echo "no CPU reading in $ERR" >&2; exit 1; fi
cpu=$(python3 -c "print($cpu_usec/1e6)")
echo "wall: $((end - start))s  cpu: ${cpu}s"
python3 score.py sample sample/answers.json "$OUT" --cpu-seconds "$cpu"

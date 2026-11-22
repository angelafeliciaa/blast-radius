#!/bin/sh
# Run the entry image on a validation set (patches + ground truth from truth.py) and score it.
# Usage: tools/validate/score_val.sh <set dir with commits/ answers.json tests.json>
set -eu
cd "$(dirname "$0")/../.."
SET=${1:?usage: score_val.sh <set dir>}
IMAGE=${IMAGE:-blast-radius-entry}
docker run --rm --platform linux/amd64 --network none --cpus 2 \
  --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp ${EXTRA_ENV:-} \
  --volume "$PWD/$SET/commits:/data/commits:ro" \
  --volume "$PWD/$SET/tests.json:/data/tests.json:ro" \
  "$IMAGE" sh -c 'python /app/selector.py; status=$?; echo "CPUSTAT $(grep usage_usec /sys/fs/cgroup/cpu.stat)" >&2; exit $status' \
  > "$SET/output.jsonl" 2> "$SET/run.err"
cpu_usec=$(grep '^CPUSTAT' "$SET/run.err" | awk '{print $3}')
python3 score.py "$SET" "$SET/answers.json" "$SET/output.jsonl" --cpu-seconds "$(python3 -c "print($cpu_usec/1e6)")"

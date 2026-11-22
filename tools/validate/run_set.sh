#!/bin/sh
# Run an image on a set of patches under the scorer's limits and score it against the set's
# answers.json: run_set.sh <image> <set dir with commits/ tests.json answers.json> [out name]
set -eu
IMAGE=$1; SET=$(cd "$2" && pwd); NAME=${3:-$1}
docker run --rm --network none --cpus 2 \
  --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
  --volume "$SET/commits:/data/commits:ro" --volume "$SET/tests.json:/data/tests.json:ro" \
  "$IMAGE" sh -c 'python /app/selector.py; status=$?; echo "CPUSTAT $(grep usage_usec /sys/fs/cgroup/cpu.stat)" >&2; exit $status' \
  > "$SET/out-$NAME.jsonl" 2> "$SET/err-$NAME.log"
cpu=$(grep '^CPUSTAT' "$SET/err-$NAME.log" | awk '{print $3/1e6}')
grep "commits in" "$SET/err-$NAME.log" || true
python3 "$(dirname "$0")/../../score.py" "$SET" "$SET/answers.json" "$SET/out-$NAME.jsonl" --cpu-seconds "$cpu"

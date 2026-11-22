#!/bin/bash
# Counts the instructions (user + kernel) and the CPU seconds of whole containers, each run
# under the scorer's flags on the sample: tools/instructions.sh <rounds> <image> [<image> ...]
#
# CPU seconds on a shared box move by 20-40% from one minute to the next; the instruction
# count of the same image repeats to about 0.1%, so it is the number to compare two builds
# by. Images are run in turn, round by round, and each one's sorted results are printed.
# Needs Linux with cgroup v2, Docker using the systemd cgroup driver, perf and sudo.
#   SET=<dir with commits/>   another set of patches instead of sample/
#   EXTRA="-e BR_SHARE=0"     extra docker run arguments
set -u
cd "$(dirname "$0")/.."
ROUNDS=${1:?usage: instructions.sh <rounds> <image>...}
shift
SET=${SET:-$PWD/sample}
SLICE=brinstr$$.slice
LOG=$(mktemp)
sudo sysctl -q -w kernel.perf_event_paranoid=-1
# every container of this script runs under one cgroup that outlives them, so its
# counters are still there to read after a container has exited
sudo mkdir -p "/sys/fs/cgroup/$SLICE/keep"
usage() { awk '/usage_usec/ {print $2}' "/sys/fs/cgroup/$SLICE/cpu.stat"; }
failed=0
for _ in $(seq 1 "$ROUNDS"); do
  for image in "$@"; do
    before=$(usage)
    sudo perf stat -a -x, -e instructions -G "$SLICE" -o "$LOG.perf" -- \
      docker run --rm --network none --cpus 2 --memory 4096m --memory-swap 4096m --pids-limit 256 \
        --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL --security-opt no-new-privileges \
        --user 65534:65534 --env HOME=/tmp --cgroup-parent="$SLICE" ${EXTRA:-} \
        --volume "$SET/commits:/data/commits:ro" --volume "$PWD/sample/tests.json:/data/tests.json:ro" \
        "$image" > "$LOG.out" 2> /dev/null
    status=$?
    sleep 0.2
    after=$(usage)
    count=$(sudo awk -F, '/instructions/ {print $1}' "$LOG.perf")
    if [ "$status" -ne 0 ] || [ -z "$count" ]; then
      echo "$image: run failed (exit $status) or perf gave no count" >&2
      failed=1
      continue
    fi
    echo "$image $(awk -v n="$count" 'BEGIN {printf "%.3f", n / 1e9}') $(awk -v a="$after" -v b="$before" 'BEGIN {printf "%.3f", (a - b) / 1e6}') $(sha256sum "$LOG.out" | cut -c1-12)" >> "$LOG"
  done
done
for image in "$@"; do
  grep -q "^$image " "$LOG" || continue
  echo "$image: instructions (G) $(grep "^$image " "$LOG" | awk '{print $2}' | sort -n | tr '\n' ' ')| cpu s $(grep "^$image " "$LOG" | awk '{print $3}' | sort -n | tr '\n' ' ')| output $(grep "^$image " "$LOG" | awk '{print $4}' | sort -u | tr '\n' ' ')"
done
sudo rm -f "$LOG" "$LOG.perf" "$LOG.out"
sudo rmdir "/sys/fs/cgroup/$SLICE/keep" "/sys/fs/cgroup/$SLICE" 2> /dev/null
exit $failed

#!/usr/bin/env bash
# Pre-launch cleanup: clear orphan Isaac / carb state left by prior runs.
# Safe to run any time — only touches files whose owning PID is no longer alive,
# and only kills processes matching narrow training-related patterns.

set -u

echo "[clean_isaac_state] starting cleanup"

# 1) Kill surviving Kit / Isaac training / ffmpeg processes owned by this user.
#    Patterns are narrow so unrelated kit processes on the box are untouched.
for pat in \
    'python .* IsaacLab/scripts/.*train\.py' \
    'python .* IsaacLab/scripts/.*play\.py' \
    'kit.*isaac-sim.*Foosball' \
    ; do
    pids=$(pgrep -u "$USER" -f "$pat" || true)
    if [ -n "$pids" ]; then
        echo "[clean_isaac_state] killing PIDs matching $pat: $pids"
        kill -9 $pids 2>/dev/null || true
    fi
done

# Orphan ffmpegs spawned by imageio for overhead-video writes.
ff_pids=$(pgrep -u "$USER" -f 'ffmpeg.*overhead-step' || true)
if [ -n "$ff_pids" ]; then
    echo "[clean_isaac_state] killing orphan ffmpeg PIDs: $ff_pids"
    kill -9 $ff_pids 2>/dev/null || true
fi

# 2) Clear orphan carb shared-memory / semaphore files in /dev/shm whose
#    owning PID is no longer alive. Match only files created by this user.
removed=0
while IFS= read -r -d '' f; do
    # extract trailing numeric PID from name (e.g. carb-RStringInternals-12345)
    base=$(basename "$f")
    pid=$(echo "$base" | grep -oE '[0-9]+$' || true)
    if [ -z "$pid" ]; then
        continue
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$f" && removed=$((removed + 1))
    fi
done < <(find /dev/shm -maxdepth 1 -user "$USER" \
    \( -name 'carb-*' -o -name 'sem.carb-*' -o -name 'kit-*' -o -name 'ov-*' \) \
    -print0 2>/dev/null)
echo "[clean_isaac_state] removed $removed orphan /dev/shm entries"

# 3) Brief wait for the GPU to actually be free of compute apps.
for i in $(seq 1 10); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$n" = "0" ]; then
        break
    fi
    sleep 1
done

echo "[clean_isaac_state] done"

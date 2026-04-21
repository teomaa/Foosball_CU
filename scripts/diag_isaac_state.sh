#!/usr/bin/env bash
# Snapshot of system state after an Isaac training run exits.
# Run this AFTER Ctrl+C on a training run (and again before the next launch if
# that one segfaults) to see what persisted across processes.

echo "=== $(date) ==="

echo "--- surviving processes (kit / carb / omniverse / train.py / ffmpeg / usd) ---"
pgrep -fa 'kit|carb|omniverse|python.*train\.py|ffmpeg|usd' || echo "(none)"

echo "--- /dev/shm entries (kit / carb / ov / isaac) ---"
ls -la /dev/shm/ 2>/dev/null | grep -iE 'kit|carb|ov|isaac' || echo "(none)"

echo "--- SysV shared-memory segments owned by $USER ---"
ipcs -m 2>/dev/null | awk -v u="$USER" 'NR<=2 || $3==u'

echo "--- SysV semaphores owned by $USER ---"
ipcs -s 2>/dev/null | awk -v u="$USER" 'NR<=2 || $3==u'

echo "--- GPU compute apps ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "(nvidia-smi not available)"

echo "--- GPU memory ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true

echo "--- recent Omniverse logs ---"
ls -lt ~/.nvidia-omniverse/logs/ 2>/dev/null | head -5 || echo "(no ~/.nvidia-omniverse/logs/)"

echo "--- recent ov cache entries ---"
ls -lt ~/.cache/ov/ 2>/dev/null | head -8 || echo "(no ~/.cache/ov/)"

echo "=== end ==="

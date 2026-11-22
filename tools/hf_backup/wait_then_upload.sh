#!/bin/bash
# wait until the given PIDs (running eval jobs) have all exited, then run the resumable backup loop
D=/root/project/evalwam/tools/hf_backup
alive() { for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
while alive "$@"; do sleep 60; done
echo "[$(date '+%m-%d %H:%M:%S')] jobs $* finished, starting upload"
HF_HUB_DISABLE_PROGRESS_BARS=1 bash $D/hf_backup_loop.sh

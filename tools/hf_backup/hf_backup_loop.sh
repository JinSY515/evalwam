#!/bin/bash
# rerun hf_backup.py every 30 min until it finishes (HF storage usage lags after deletes + history squash)
D=/root/project/evalwam/tools/hf_backup
for i in $(seq 1 48); do
  echo "=== attempt $i $(date '+%m-%d %H:%M')"
  HF_HUB_DISABLE_PROGRESS_BARS=1 /venv/main/bin/python $D/hf_backup.py && { echo "=== loop done"; exit 0; }
  sleep 1800
done
echo "=== loop gave up after 48 attempts"

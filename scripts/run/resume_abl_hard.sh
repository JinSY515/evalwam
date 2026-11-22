#!/bin/bash
# Resume an interrupted ablation on 'hard' (infer_cosmos_lora skips existing sample_XX.mp4, keeps idx),
# then restore goal_frame for the skipped samples -- the resumed run rewrites samples_meta.json and
# drops goal_frame for every sample it skips.   $1 id  $2 gpu  $3 run subdir
H=/root/project/egowam/results_syn/abl/$3/valbench5s/hard
bash /root/project/evalwam/scripts/run/run_abl.sh $1 hard $2 > /root/project/egowam/results_syn/abl/${1}_hard_resume.log 2>&1 || echo "FAIL $1 hard"
python3 - "$H" <<'PY'
import json, sys
H = sys.argv[1]
old = json.load(open(f"{H}/samples_meta.before_resume.json")); new = json.load(open(f"{H}/samples_meta.json"))
n = 0
for k, v in old.items():
    if "goal_frame" in v and "goal_frame" not in new.get(k, {}):
        assert new[k]["seq_dir"] == v["seq_dir"] and new[k]["start_frame"] == v["start_frame"], k
        new[k]["goal_frame"] = v["goal_frame"]; n += 1
json.dump(new, open(f"{H}/samples_meta.json", "w"), indent=2)
print(f"restored goal_frame for {n} samples; total with goal_frame:", sum("goal_frame" in v for v in new.values()), "/", len(new))
PY
echo "ABL_${1}_DONE"

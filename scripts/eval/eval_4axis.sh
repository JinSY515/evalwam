#!/bin/bash
# 하나의 추론 산출물을 네 축으로 평가한다. 추론은 하지 않으므로 몇 번이고 재실행 가능.
# 축마다 auto(기본, 이미 N행이면 건너뜀) | 1(다시) | 0(생략).
#
#   #2 VQ     tools/eval_video_quality.py    -> video_quality_metrics.csv   psnr/ssim/lpips
#   #3 CHAN   tools/eval_channel_vs_da3.py   -> channel_vs_da3_metrics.csv  영상<->action(raymap)
#   #1 CAM    tools/eval_camera_control.py   -> camera_control_metrics.csv  ATE/RPE
#   #4 DEPTH  tools/eval_depth_vs_da3.py     -> depth_vs_da3_metrics.csv    영상<->depth
#
# 이 런의 "action" 은 raymap 이다. --pred_action 헤드가 없고 _pred_actions.npy 가 0개이며
# _action_poses.npy 는 GT 와 max|diff|=0 (= 예측이 아니라 GT 트랙) 이므로
# eval_channel_vs_da3 는 --channel raymap 으로만 의미가 있다.
#
# Usage:
#   TRAIN_DIR=$RS/goalprog_appearance_depth_fine CKPT=checkpoint-3000 GPU=0 bash <this>
#   D=<pred_depth dir> bash <this>
#   CAM=0 DEPTH=0 bash <this>      # 코드 변경 0 으로 도는 범위만
#   VQ=1 CHAN=1 CAM=1 bash <this>  # 전부 다시

COSMOS=${COSMOS:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
RS=${RS:-/root/project/ego_wam/results_syn}
TAG=${TAG:-valbench5s}
PY=${PY:-/root/miniconda3/envs/eval/bin/python}
GLDIR=${GLDIR:-/root/project/egowm/_gl}
GPU=${GPU:-0}
NUM_FRAMES=${NUM_FRAMES:-77}

if [ -z "${D:-}" ]; then
  [ -n "${TRAIN_DIR:-}" ] || { echo "[e4] TRAIN_DIR 또는 D 가 필요하다"; exit 1; }
  D="$TRAIN_DIR/inference-f77/$TAG/${CKPT:-checkpoint-3000}/pred_depth"
fi

VQ=${VQ:-auto}; CHAN=${CHAN:-auto}; CAM=${CAM:-auto}; DEPTH=${DEPTH:-auto}
DEPTH_FLOOR=${DEPTH_FLOOR:-1}      # C 의 바닥. A 는 바닥이 아니다 (영상 이탈이 섞인다)
DA3_SUB=${DA3_SUB:-viz_da3_fix}
FLOOR_SRC=${FLOOR_SRC:-$RS/abl_interpolate_hist_goal_predaction/inference-f77/checkpoint-8000/valbench5s/da3_floor_metrics.csv}

# ---- 0. 사전점검. 반쯤 만들어진 CSV 보다 여기서 멈추는 게 낫다 (여기서만 exit) ----
[ -d "$D" ] || { echo "[e4] missing D $D"; exit 1; }
[ -f "$D/samples_meta.json" ] || { echo "[e4] missing $D/samples_meta.json"; exit 1; }
for t in tools/eval_video_quality.py tools/eval_channel_vs_da3.py tools/eval_camera_control.py; do
  [ -f "$COSMOS/$t" ] || { echo "[e4] missing $t"; exit 1; }
done
N=$("$PY" -c "import json;print(len(json.load(open('$D/samples_meta.json'))))")
echo "[e4] D=$D"
echo "[e4] N=$N  VQ=$VQ CHAN=$CHAN CAM=$CAM DEPTH=$DEPTH(floor=$DEPTH_FLOOR)"

rows() { [ -f "$1" ] && "$PY" -c "
import csv,sys
try: print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))
except Exception: print(0)" "$1" || echo 0; }

# auto: 행수가 정확히 N 일 때만 건너뛴다. 부분 CSV 는 재사용하지 않는다 --
# eval_camera_control.py:282 가 증분 저장하되 '이번 런의 rows 만' 쓰므로 이어붙지 않는다.
todo() {  # todo <MODE> <csv>
  [ "$1" = "0" ] && return 1
  [ "$1" = "1" ] && return 0
  [ "$(rows "$2")" = "$N" ] && return 1 || return 0
}

# ---- #4 패치 미적용이면 그 축만 끈다 (런처가 패치 순서에 묶이지 않게) ----
if [ "$DEPTH" != "0" ] && ! grep -q -- "--save_depth" "$COSMOS/tools/viz_da3_pointcloud.py"; then
  echo "[e4] [note] viz_da3_pointcloud.py has no --save_depth -> skipping #4"; DEPTH=0
fi
if [ "$DEPTH" != "0" ] && [ ! -f "$COSMOS/tools/eval_depth_vs_da3.py" ]; then
  echo "[e4] [note] tools/eval_depth_vs_da3.py missing -> skipping #4"; DEPTH=0
fi
if [ "$DEPTH" != "0" ] && [ "$DEPTH_FLOOR" != "0" ] \
   && ! grep -q -- "--da3_on_gt" "$COSMOS/tools/viz_da3_pointcloud.py"; then
  echo "[e4] [note] no --da3_on_gt -> skipping F; C would then have no baseline to read against"; DEPTH_FLOOR=0
fi

cd "$COSMOS" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
export LD_LIBRARY_PATH="$GLDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- 1. #2 video quality. 제일 싸고, meta 버그를 GPU 태우기 전에 드러낸다 ----
VQ_CSV="$D/video_quality_metrics.csv"
if todo "$VQ" "$VQ_CSV"; then
  echo "[e4] #2 video quality ..."
  "$PY" tools/eval_video_quality.py --results_dir "$D" --out_csv "$VQ_CSV" \
    || echo "[e4]   [fail] video quality"
else
  echo "[e4] #2 video quality: $(rows "$VQ_CSV") rows present -> skip"
fi
BADN=$("$PY" -c "
import csv,os
p='$VQ_CSV'
print(0 if not os.path.exists(p) else
      sum(1 for r in csv.DictReader(open(p)) if int(r.get('n',0))!=$NUM_FRAMES))")
if [ "$BADN" != "0" ]; then
  echo "[e4] !! n != $NUM_FRAMES : ${BADN} rows -> samples_meta bug. disabling GPU stages"
  CHAN=0; CAM=0; DEPTH=0
fi

# ---- 1b. DA3 on the GENERATED video. One pass yields BOTH _da3_poses.npy (what #3 reads)
# and _da3_depth.npy (what #4 reads), so it has to come before #3 -- a checkpoint whose
# viz_da3_fix/ is empty would otherwise make #3 skip all N samples in silence.
NDP=$(ls "$D/$DA3_SUB"/*_da3_poses.npy 2>/dev/null | wc -l | tr -d ' ')
NDD=$(ls "$D/$DA3_SUB"/*_da3_depth.npy 2>/dev/null | wc -l | tr -d ' ')
NEED_DA3=0
[ "$CHAN" != "0" ] && [ "$NDP" != "$N" ] && NEED_DA3=1
[ "$DEPTH" != "0" ] && [ "$NDD" != "$N" ] && NEED_DA3=1
if [ "$NEED_DA3" = "1" ]; then
  SDARG=(); [ "$DEPTH" != "0" ] && SDARG=(--save_depth)
  echo "[e4] #1b DA3 on generated video (poses=$NDP/$N depth=$NDD/$N) ..."
  "$PY" tools/viz_da3_pointcloud.py --results_dir "$D" --source da3 \
      --out_subdir "$DA3_SUB" "${SDARG[@]}" \
    || echo "[e4]   [fail] da3 pass (generated)"
else
  echo "[e4] #1b DA3 generated: poses=$NDP depth=$NDD -> skip"
fi
if [ "$CHAN" != "0" ] \
   && [ "$(ls "$D/$DA3_SUB"/*_da3_poses.npy 2>/dev/null | wc -l | tr -d ' ')" != "$N" ]; then
  echo "[e4] !! da3_poses != N -> #3 would silently skip every sample. disabling #3"
  CHAN=0
fi

# ---- 2. #3 action(=raymap) <-> 영상. 기존 DA3 포즈를 읽으므로 GPU 0 ----
CH_CSV="$D/channel_vs_da3_metrics.csv"
if [ "$CHAN" != "0" ]; then
  # floor 를 맹목 복사하지 않는다: (seq_dir, start_frame, num_frames, frame_stride) 가 같을 때만.
  if [ ! -f "$D/da3_floor_metrics.csv" ] && [ -f "$FLOOR_SRC" ]; then
    SAME=$("$PY" -c "
import json,os
def key(p):
    m=json.load(open(p))
    return [(m[k]['seq_dir'],int(m[k]['start_frame']),int(m[k]['num_frames']),
             int(m[k].get('frame_stride',1))) for k in sorted(m,key=int)]
src=os.path.join(os.path.dirname('$FLOOR_SRC'),'samples_meta.json')
print('1' if os.path.exists(src) and key(src)==key('$D/samples_meta.json') else '0')")
    if [ "$SAME" = "1" ]; then
      cp "$FLOOR_SRC" "$D/da3_floor_metrics.csv"; echo "[e4]   reusing floor (meta verified identical)"
    else
      echo "[e4]   [warn] FLOOR_SRC meta differs -> not copied. run da3_floor.py for this dir"
    fi
  fi
  [ -f "$CH_CSV" ] && [ ! -f "$CH_CSV.bak_prefloor" ] && cp "$CH_CSV" "$CH_CSV.bak_prefloor"
  echo "[e4] #3 channel(raymap) <-> DA3 ..."
  # out_csv/floor_csv 는 results_dir 과 join 된다 (eval_channel_vs_da3.py:167,:86) -> 파일명만.
  "$PY" tools/eval_channel_vs_da3.py --results_dir "$D" --channel raymap \
      --da3_poses_dir "$DA3_SUB" --floor_csv da3_floor_metrics.csv \
      --out_csv channel_vs_da3_metrics.csv \
    || echo "[e4]   [fail] channel"
else
  echo "[e4] #3 channel: skipped"
fi

# ---- 3. #1 camera controllability. --estimators 기본이 mast3r,colmap 이라 da3 를 명시 ----
CAM_CSV="$D/camera_control_metrics.csv"
if todo "$CAM" "$CAM_CSV"; then
  echo "[e4] #1 camera control (DA3, ${N} samples) ..."
  "$PY" tools/eval_camera_control.py --results_dir "$D" --estimators da3 \
      --out_csv "$CAM_CSV" \
    || echo "[e4]   [fail] camera control"
else
  echo "[e4] #1 camera control: $(rows "$CAM_CSV") rows present -> skip"
fi

# ---- 4. #4 depth <-> 영상 ----
DP_CSV="$D/depth_vs_da3_metrics.csv"
if [ "$DEPTH" != "0" ]; then
  if [ "$DEPTH_FLOOR" != "0" ]; then
    NDG=$(ls "$D/$DA3_SUB"/*_da3_gt_depth.npy 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NDG" != "$N" ]; then
      echo "[e4] #4b DA3 depth (GT video = floor F) ..."
      "$PY" tools/viz_da3_pointcloud.py --results_dir "$D" --source da3 \
          --out_subdir "$DA3_SUB" --save_depth --da3_on_gt \
        || echo "[e4]   [fail] da3 depth (gt)"
    else
      echo "[e4] #4b DA3 depth (GT): ${NDG} present -> skip"
    fi
  fi
  if todo "$DEPTH" "$DP_CSV"; then
    echo "[e4] #4c depth compare ..."
    "$PY" tools/eval_depth_vs_da3.py --results_dir "$D" --da3_subdir "$DA3_SUB" \
        --out_csv depth_vs_da3_metrics.csv \
      || echo "[e4]   [fail] depth compare"
  else
    echo "[e4] #4c depth compare: $(rows "$DP_CSV") rows present -> skip"
  fi
else
  echo "[e4] #4 depth: skipped"
fi

# ---- 5. 요약. floor 는 반드시 같은 줄에 (따로 두면 안 보게 된다) ----
echo; echo "[e4] ======== RESULT  ($(date +%H:%M:%S))"
# Summary is tools/report_4axis.py -- one reporter, used here and standalone, so the two
# can never drift apart. It only reads CSVs, so it is safe even if a stage above failed.
if [ -f "$COSMOS/tools/report_4axis.py" ]; then
  "$PY" "$COSMOS/tools/report_4axis.py" --dirs "$D"
else
  echo "[e4] (tools/report_4axis.py missing -- listing CSV row counts instead)"
  for f in video_quality channel_vs_da3 camera_control depth_vs_da3; do
    printf "   %-22s %s rows\n" "$f" "$(rows "$D/${f}_metrics.csv")"
  done
fi

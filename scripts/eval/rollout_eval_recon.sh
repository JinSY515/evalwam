#!/bin/bash
# rollout 산출물 하나를 끝까지 후처리한다. 추론은 하지 않으므로 몇 번이고 재실행 가능.
# 세 단계 모두 "이미 된 건 건너뛴다" 가 기본이고 의미론이 대칭이다:
#   POSES / EVAL = auto(기본, 있는 건 재사용) | 1(전부 다시) | 0(생략)
#   SKIP_EXISTING = orbit 재렌더 제어 (포즈가 mp4 보다 새로우면 자동 해제)
#
#   0) tools/invert_raymap_rollout.py  (GPU, VAE만)
#        청크별 _pred_raymap 역변환 + 체이닝 -> rollout_NN_pred_poses.npy (381,6)
#   1) tools/eval_rollout.py           -> rollout_metrics.csv + viz/rollout_NN_vs_gt.mp4
#        (GT | Rollout | Goal | Depth 패널)
#   2) tools/recon_orbit_rollout.py    -> recon/orbit_rollout_sNN_<color>.mp4
#        포즈 없으면 2패널, 있으면 4패널
#
# VIZ_DEPTH_MODE 기본 'si': 우리 DepthUpHead 출력은 metric 미터가 아니라 SI-disparity
# (실측 0.036~0.150) 라서 metric 스케일([0,20] m)로 칠하면 median 이 0.7/255 가 되어
# 패널 전체가 한 색이 된다. metric 헤드를 볼 때만 metric 으로.
#
# Usage:
#   TRAIN_DIR=.../goalprog_appearance_depth_fine CKPT=checkpoint-3500 GPU=0 bash <this>
#   ROLL_DIR=<rollout dir> bash <this>
#   EVAL=1 POSES=1 SKIP_EXISTING=0 bash <this>    # 전부 처음부터
#   EVAL=0 SAMPLES="0 1 2" bash <this>            # orbit 만, 3샘플
#   NO_QUALITY=1 bash <this>                      # PSNR/LPIPS 생략 (빠름)
set -uo pipefail

COSMOS=${COSMOS:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
RS=${RS:-/root/project/ego_wam/results_syn}
TAG=${TAG:-valbench}
NUM_FRAMES=${NUM_FRAMES:-77}
if [ -n "${ROLL_DIR:-}" ]; then
  D="$ROLL_DIR"
else
  TRAIN_DIR=${TRAIN_DIR:-$RS/goalprog_appearance_depth_fine}
  CKPT_ARG=${CKPT:-latest}
  if [ "$CKPT_ARG" = "latest" ]; then
    CKPT_NAME=$(basename "$(ls -dt "$TRAIN_DIR/rollout-f${NUM_FRAMES}/${TAG}"/checkpoint-* 2>/dev/null | head -1)")
  else
    CKPT_NAME="$CKPT_ARG"
  fi
  D="$TRAIN_DIR/rollout-f${NUM_FRAMES}/${TAG}/$CKPT_NAME"
fi

POSES=${POSES:-auto}
POSE_ANCHOR=${POSE_ANCHOR:-chain}
POSE_SKIP=${POSE_SKIP:-1}
[ "$POSES" = "1" ] && POSE_SKIP=0   # POSES=1 은 "강제 재생성" 이므로 skip 과 모순
EVAL=${EVAL:-auto}
EVAL_SKIP=1
[ "$EVAL" = "1" ] && EVAL_SKIP=0    # EVAL=1 은 "전부 다시" 이므로 skip 과 모순
RECON=${RECON:-1}
BEV=${BEV:-1}                  # orbit 과 별도로 BEV PNG 도 만든다 (탐험 면적 m^2 포함)
AREA_MIN_PTS=${AREA_MIN_PTS:-3}
VIZ_DEPTH_MODE=${VIZ_DEPTH_MODE:-si}
SKIP_EXISTING=${SKIP_EXISTING:-1}
ORBIT_COLOR=${ORBIT_COLOR:-rgb}
ORBIT_SIZE=${ORBIT_SIZE:-400}
ORBIT_FRAMES=${ORBIT_FRAMES:-72}
MAX_DEPTH=${MAX_DEPTH:-35}
STRIDE=${STRIDE:-6}
EXTENT=${EXTENT:-0}            # 0 = GT 구름 percentile 자동
EXTENT_PCT=${EXTENT_PCT:-98}
RAY_DS=${RAY_DS:-2}
MOMENT_SCALE=${MOMENT_SCALE:-8.0}

n_poses () { ls "$D"/rollout_*_pred_poses.npy 2>/dev/null | wc -l | tr -d ' '; }

# ---- 사전 점검 (반쯤 만들어진 산출물보다 여기서 멈추는 게 낫다) ----
[ -d "$D" ] || { echo "[rer] missing ROLL_DIR $D"; exit 1; }
[ -f "$D/rollout_meta.json" ] || { echo "[rer] missing $D/rollout_meta.json"; exit 1; }
for t in tools/eval_rollout.py tools/recon_orbit_rollout.py tools/invert_raymap_rollout.py; do
  [ -f "$COSMOS/$t" ] || { echo "[rer] missing $t"; exit 1; }
done
if [ "$EVAL" != "0" ] && [ "$VIZ_DEPTH_MODE" = "si" ]; then
  grep -q -- "--viz_depth_mode" "$COSMOS/tools/eval_rollout.py" \
    || { echo "[rer] eval_rollout.py has no --viz_depth_mode: apply the patch, or VIZ_DEPTH_MODE=metric"; exit 1; }
fi

# --skip_existing 은 순수 최적화다. 툴에 없으면 멈추지 말고 "전부 재계산" 으로 내려간다
# (느릴 뿐 결과는 같다) -- 그래야 이 런처가 툴 패치 적용 순서에 묶이지 않는다.
if [ "$EVAL_SKIP" != "0" ] \
   && ! grep -q -- "--skip_existing" "$COSMOS/tools/eval_rollout.py"; then
  echo "[rer] [note] eval_rollout.py 에 --skip_existing 이 없다 -> eval 전부 재계산"
  EVAL_SKIP=0
fi
if [ "$POSE_SKIP" != "0" ] \
   && ! grep -q -- "--skip_existing" "$COSMOS/tools/invert_raymap_rollout.py"; then
  echo "[rer] [note] invert_raymap_rollout.py 에 --skip_existing 이 없다 -> 포즈 전부 재계산"
  POSE_SKIP=0
fi

N_DEPTH=$(ls "$D"/rollout_*_pred_depth.npy 2>/dev/null | wc -l | tr -d ' ')
N_RAYMAP=$(ls "$D"/rollout_*_pred_raymap.npy 2>/dev/null | wc -l | tr -d ' ')
P_BEFORE=$(n_poses)
if [ -z "${SAMPLES:-}" ]; then
  SAMPLES=$(python3 -c "
import json
m=json.load(open('$D/rollout_meta.json'))
print(' '.join(sorted(m, key=lambda x:int(x))))")
fi
NS=$(echo $SAMPLES | wc -w | tr -d ' ')

echo "[rer] ROLL_DIR=$D"
echo "[rer] 샘플 ${NS}개: $SAMPLES"
echo "[rer] pred_depth=${N_DEPTH}  pred_raymap=${N_RAYMAP}  pred_poses=${P_BEFORE}"
echo "[rer] POSES=$POSES(anchor=$POSE_ANCHOR skip=$POSE_SKIP)  EVAL=$EVAL(depth=$VIZ_DEPTH_MODE skip=$EVAL_SKIP)  RECON=$RECON(color=$ORBIT_COLOR)  BEV=$BEV(min_pts=$AREA_MIN_PTS)"
python3 -c "
import json;m=json.load(open('$D/rollout_meta.json'))
print('[rer] num_chunks=%s num_frames=%s' % (sorted({m[k]['num_chunks'] for k in m}),
                                             sorted({m[k]['num_frames'] for k in m})))"

# ---- 0) 포즈 ----
RUN_POSES=0
if [ "$POSES" = "1" ]; then
  RUN_POSES=1
elif [ "$POSES" = "auto" ] && [ "${P_BEFORE}" -lt "${NS}" ]; then
  RUN_POSES=1
fi
if [ "$RUN_POSES" = "1" ]; then
  if [ "${N_RAYMAP}" -eq 0 ]; then
    echo "[rer] 0) skip: _pred_raymap.npy 가 없다 (--pred_raymap 없이 추론한 rollout)"
  else
    echo; echo "======== 0) invert_raymap_rollout  ($(date +%H:%M:%S))"
    PSKIP=""; [ "$POSE_SKIP" != "0" ] && PSKIP="--skip_existing"
    ( cd "$COSMOS" && CUDA_VISIBLE_DEVICES=${GPU:-0} python3 tools/invert_raymap_rollout.py \
        --roll_dir "$D" --anchor "$POSE_ANCHOR" $PSKIP \
        --ray_downsample "$RAY_DS" --raymap_moment_scale "$MOMENT_SCALE" ) \
      || echo "[rer] invert_raymap_rollout FAILED -- 2패널로 계속한다"
  fi
else
  echo "[rer] 0) 건너뜀 (POSES=$POSES, 기존 포즈 ${P_BEFORE}개)"
fi
P_AFTER=$(n_poses)
if [ "${P_AFTER}" -gt "${P_BEFORE}" ]; then
  echo "[rer] 포즈 ${P_BEFORE} -> ${P_AFTER} 개  =>  orbit 을 4패널로 재렌더 (SKIP_EXISTING 해제)"
  SKIP_EXISTING=0
fi
if [ "${P_AFTER}" -eq 0 ]; then
  echo "[rer] 포즈 0/${NS} 샘플  ->  orbit 전부 2패널"
elif [ "${P_AFTER}" -lt "${NS}" ]; then
  echo "[rer] 포즈 ${P_AFTER}/${NS} 샘플  ->  그 ${P_AFTER}개는 4패널, 나머지 $((NS-P_AFTER))개는 2패널"
else
  echo "[rer] 포즈 ${P_AFTER}/${NS} 샘플  ->  orbit 전부 4패널"
fi

# ---- 1) eval + viz ----
if [ "$EVAL" != "0" ]; then
  echo; echo "======== 1) eval_rollout  ($(date +%H:%M:%S))"
  ESKIP=""; [ "$EVAL_SKIP" != "0" ] && ESKIP="--skip_existing"
  ( cd "$COSMOS" && python3 tools/eval_rollout.py \
      --roll_dir "$D" --viz --viz_depth_mode "$VIZ_DEPTH_MODE" $ESKIP \
      ${NO_QUALITY:+--no_quality} ${DRIFT:+--drift} ) \
    || echo "[rer] eval_rollout FAILED -- recon 은 계속한다"
fi

# ---- 2) orbit 리컨 ----
if [ "$RECON" != "0" ]; then
  echo; echo "======== 2) recon_orbit_rollout  ($(date +%H:%M:%S))"
  mkdir -p "$D/recon"
  made=0; skipped=0; failed=0
  for S in $SAMPLES; do
    OUT="$D/recon/orbit_rollout_s$(printf %02d "$S")_${ORBIT_COLOR}.mp4"
    PP="$D/rollout_$(printf %02d "$S")_pred_poses.npy"
    STALE=0
    [ -f "$PP" ] && [ -s "$OUT" ] && [ "$PP" -nt "$OUT" ] && STALE=1   # 포즈가 더 새롭다 = 2패널 잔재
    if [ "$SKIP_EXISTING" != "0" ] && [ "$STALE" = "0" ] && [ -s "$OUT" ] \
       && [ "$(stat -c%s "$OUT" 2>/dev/null || echo 0)" -gt 20000 ]; then
      skipped=$((skipped+1)); continue
    fi
    [ "$STALE" = "1" ] && echo "[rer]   sample $S: 포즈가 mp4 보다 새롭다 -> 4패널로 재렌더"
    if ( cd "$COSMOS" && python3 tools/recon_orbit_rollout.py \
           --roll_dir "$D" --sample "$S" --out "$OUT" \
           --color "$ORBIT_COLOR" --size "$ORBIT_SIZE" --frames "$ORBIT_FRAMES" \
           --max_depth "$MAX_DEPTH" --stride "$STRIDE" \
           --extent "$EXTENT" --extent_pct "$EXTENT_PCT" ); then
      made=$((made+1))
    else
      failed=$((failed+1)); echo "[rer]   [fail] sample $S"
    fi
  done
  echo "[rer] orbit: 생성 ${made} / 건너뜀 ${skipped} / 실패 ${failed}"
fi

# ---- 3) BEV 리컨 (PNG. 탐험 면적 m^2 를 패널마다 찍는다) ----
if [ "$BEV" != "0" ]; then
  echo; echo "======== 3) recon_orbit_rollout --view bev  ($(date +%H:%M:%S))"
  mkdir -p "$D/recon"
  bmade=0; bskip=0; bfail=0
  for S in $SAMPLES; do
    BOUT="$D/recon/bev_rollout_s$(printf %02d "$S").png"
    PP="$D/rollout_$(printf %02d "$S")_pred_poses.npy"
    BSTALE=0
    [ -f "$PP" ] && [ -s "$BOUT" ] && [ "$PP" -nt "$BOUT" ] && BSTALE=1
    if [ "$SKIP_EXISTING" != "0" ] && [ "$BSTALE" = "0" ] && [ -s "$BOUT" ] \
       && [ "$(stat -c%s "$BOUT" 2>/dev/null || echo 0)" -gt 5000 ]; then
      bskip=$((bskip+1)); continue
    fi
    [ "$BSTALE" = "1" ] && echo "[rer]   sample $S: 포즈가 png 보다 새롭다 -> 4패널로 재렌더"
    if ( cd "$COSMOS" && python3 tools/recon_orbit_rollout.py \
           --roll_dir "$D" --sample "$S" --out "$BOUT" --view bev \
           --color "$ORBIT_COLOR" --size "$ORBIT_SIZE" \
           --max_depth "$MAX_DEPTH" --stride "$STRIDE" \
           --extent "$EXTENT" --extent_pct "$EXTENT_PCT" \
           --area_min_pts "$AREA_MIN_PTS" ); then
      bmade=$((bmade+1))
    else
      bfail=$((bfail+1)); echo "[rer]   [fail] bev sample $S"
    fi
  done
  echo "[rer] bev: 생성 ${bmade} / 건너뜀 ${bskip} / 실패 ${bfail}"
fi

echo; echo "======== 결과  ($(date +%H:%M:%S))"
echo "  poses   : ${P_AFTER}개  + $D/rollout_pose_metrics.csv"
echo "  metrics : $D/rollout_metrics.csv"
echo "  viz     : $D/viz/      ($([ -d "$D/viz" ] && ls "$D/viz" 2>/dev/null | wc -l | tr -d ' ' || echo 0) 파일)"
echo "  orbit   : $D/recon/    ($([ -d "$D/recon" ] && ls "$D/recon"/*.mp4 2>/dev/null | wc -l | tr -d ' ' || echo 0) mp4)"
echo "  bev     : $D/recon/    ($([ -d "$D/recon" ] && ls "$D/recon"/bev_*.png 2>/dev/null | wc -l | tr -d ' ' || echo 0) png)"

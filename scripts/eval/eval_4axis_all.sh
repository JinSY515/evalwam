#!/bin/bash
# eval_4axis.sh 를 여러 체크포인트에 돌린다. GPU 하나당 한 체크포인트, 동시에 NGPU 개.
#
# 왜 별도 파일인가: eval_4axis.sh 는 "산출물 하나를 평가한다" 로 끝까지 단순해야
# 재실행·부분실행 의미론이 깨지지 않는다. 여러 개를 도는 것은 그 위의 스케줄링이다.
#
# Usage:
#   bash eval_4axis_all.sh                       # fine + fine_geo 의 완결된 ckpt 전부, GPU 0,1
#   GPUS="0 1 2 3" bash eval_4axis_all.sh        # 4개 동시
#   RUNS=goalprog_appearance_depth_fine bash eval_4axis_all.sh
#   CKPTS="checkpoint-3000 checkpoint-3500" bash eval_4axis_all.sh
#   DRY=1 bash eval_4axis_all.sh                 # 무엇이 돌지만 출력
#   CAM=0 DEPTH=0 bash eval_4axis_all.sh         # 하위 런처로 그대로 전달

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RS=${RS:-/root/project/ego_wam/results_syn}
TAG=${TAG:-valbench5s}
PY=${PY:-/root/miniconda3/envs/eval/bin/python}
GPUS=${GPUS:-"0 1"}
RUNS=${RUNS:-"goalprog_appearance_depth_fine goalprog_appearance_depth_fine_geo"}
DRY=${DRY:-0}

[ -f "$HERE/eval_4axis.sh" ] || { echo "[all] missing $HERE/eval_4axis.sh"; exit 1; }

# ---- 1. 후보 수집. 추론이 덜 끝난 디렉터리는 **여기서 떨군다** ----
# meta 는 120을 말하는데 pred_depth 가 15개뿐인 경우가 실제로 있다 (fine/ckpt-6000, 진행 중).
# 그대로 돌리면 모든 툴이 나머지를 조용히 skip 하고 부분 CSV 가 남는다.
JOBS=()
for R in $RUNS; do
  for P in "$RS/$R"/inference-f77/"$TAG"/checkpoint-*/pred_depth; do
    [ -d "$P" ] || continue
    C=$(basename "$(dirname "$P")")
    if [ -n "${CKPTS:-}" ]; then
      case " $CKPTS " in *" $C "*) ;; *) continue ;; esac
    fi
    [ -f "$P/samples_meta.json" ] || { echo "[all] skip $R/$C: samples_meta.json 없음"; continue; }
    NM=$("$PY" -c "import json;print(len(json.load(open('$P/samples_meta.json'))))" 2>/dev/null)
    ND=$(ls "$P"/sample_*_pred_depth.npy 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NM" != "$ND" ]; then
      echo "[all] skip $R/$C: inference incomplete (pred_depth $ND / meta $NM)"
      continue
    fi
    # eval_camera_control.py:282 saves incrementally, so two writers do not error --
    # they interleave silently. A recently touched CSV means someone is in there.
    if [ -n "$(find "$P" -maxdepth 1 -name '*_metrics.csv' -newermt "-${LOCK_SEC:-300} seconds" 2>/dev/null)" ]; then
      echo "[all] skip $R/$C: metrics touched recently -- another process looks active (LOCK_SEC=0 to ignore)"
      continue
    fi
    JOBS+=("$R|$C|$P")
  done
done
NJ=${#JOBS[@]}
NG=$(echo $GPUS | wc -w | tr -d ' ')
echo "[all] ${NJ} target(s) / ${NG} gpu(s) ($GPUS)"
for J in "${JOBS[@]}"; do echo "[all]   ${J%%|*}  $(echo "$J" | cut -d'|' -f2)"; done
[ "$NJ" = "0" ] && { echo "[all] nothing to run"; exit 0; }
[ "$DRY" != "0" ] && { echo "[all] DRY=1 -> 종료"; exit 0; }

# ---- 2. GPU 수만큼 묶어서 실행. nohup 없음: & + wait 로 포그라운드에 묶어둔다 ----
GARR=($GPUS)
i=0; WAVE=0
while [ $i -lt $NJ ]; do
  WAVE=$((WAVE+1)); PIDS=(); LABELS=()
  for g in "${GARR[@]}"; do
    [ $i -ge $NJ ] && break
    J="${JOBS[$i]}"; R="${J%%|*}"; C=$(echo "$J" | cut -d'|' -f2); P="${J##*|}"
    LOG="$P/eval_4axis.log"
    echo "[all] wave $WAVE  gpu $g  <- $R/$C   (log: $LOG)"
    ( GPU="$g" D="$P" PY="$PY" RS="$RS" TAG="$TAG" \
      VQ="${VQ:-auto}" CHAN="${CHAN:-auto}" CAM="${CAM:-auto}" \
      DEPTH="${DEPTH:-auto}" DEPTH_FLOOR="${DEPTH_FLOOR:-1}" \
      bash "$HERE/eval_4axis.sh" ) > "$LOG" 2>&1 &
    PIDS+=($!); LABELS+=("$R/$C"); i=$((i+1))
  done
  # 한 wave 가 끝나야 다음 wave 가 GPU 를 쓴다. 실패해도 멈추지 않는다.
  for k in "${!PIDS[@]}"; do
    if wait "${PIDS[$k]}"; then echo "[all]   [ok]   ${LABELS[$k]}"
    else                        echo "[all]   [fail] ${LABELS[$k]}  -> see the log"; fi
  done
done

# ---- 3. 전체 표. 축별 행수와 핵심 median 을 한 줄에 ----
echo; echo "[all] ======== OVERALL  ($(date +%H:%M:%S))"
for J in "${JOBS[@]}"; do
  R="${J%%|*}"; C=$(echo "$J" | cut -d'|' -f2); P="${J##*|}"
  echo "--- $R / $C"
  sed -n '/======== 결과/,$p' "$P/eval_4axis.log" 2>/dev/null | tail -n +2 \
    || echo "   (no log)"
done

#!/usr/bin/env bash
# Run camera-control + image-quality (+ optional MEt3R) evals on one inference results dir.
#
# Usage:   bash eval.sh <results_dir> [variant]
#   <results_dir> = inference out_dir, must contain:
#     - sample_*.mp4          (generated videos; *_vis.mp4 are ignored)
#     - samples_meta.json     (GT mapping for image quality)
#     - poses.npy             (GT trajectory for camera control)
#   [variant] = map | nomap | both     [map]
#     Worldmap inference emits BOTH sample_XX.mp4 and sample_XX_nomap.mp4 from one run
#     (same latents, same seed, map removed on the second pass). They are scored against the
#     SAME ground truth, so the difference between the two CSVs is the map's doing.
#       map    sample_XX.mp4        -> camera_control_metrics.csv       (unchanged names)
#       nomap  sample_XX_nomap.mp4  -> camera_control_metrics_nomap.csv
#       both   runs each, then prints a map-vs-nomap delta table
#
# Env toggles (defaults in []):
#   RUN_CAMERA=1  RUN_QUALITY=1  RUN_MET3R=0      # met3r is optional (needs palindrome videos)
#   VARIANT=map                                   # same as the positional [variant]
#   ESTIMATORS=mast3r,colmap                       # camera pose estimators
#   LPIPS_NET=alex   MET3R_BATCH=1   MET3R_SCRIPT=<path to eval_met3r.py>
#
# Outputs (written into <results_dir>):
#   camera_control_metrics[_nomap].csv   video_quality_metrics[_nomap].csv
#   met3r_results[_nomap].json
set -uo pipefail

RESULTS_DIR="${1:-}"
if [[ -z "$RESULTS_DIR" ]]; then
  echo "usage: bash eval.sh <results_dir> [map|nomap|both]" >&2; exit 2
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "[err] not a directory: $RESULTS_DIR" >&2; exit 2
fi

# Positional wins over the env var so `bash eval.sh <dir> nomap` always does what it says.
VARIANT="${2:-${VARIANT:-map}}"
case "$VARIANT" in
  map)   VARIANTS=(map) ;;
  nomap) VARIANTS=(nomap) ;;
  both)  VARIANTS=(map nomap) ;;
  *) echo "[err] variant must be map|nomap|both (got '$VARIANT')" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS="$(cd "$SCRIPT_DIR/../.." && pwd)"
CAM="$COSMOS/tools/eval_camera_control.py"
QUAL="$COSMOS/tools/eval_video_quality.py"
MET3R="${MET3R_SCRIPT:-$SCRIPT_DIR/bef/eval_met3r.py}"

RUN_CAMERA="${RUN_CAMERA:-1}"
RUN_QUALITY="${RUN_QUALITY:-1}"
RUN_MET3R="${RUN_MET3R:-0}"
ESTIMATORS="${ESTIMATORS:-mast3r,colmap}"
LPIPS_NET="${LPIPS_NET:-alex}"
MET3R_BATCH="${MET3R_BATCH:-1}"
PY="${PYTHON:-python3}"

# MASt3R = cloned repo (not pip-installed) + metric ckpt. Export BEFORE camera eval so
# eval_camera_control.py can import mast3r/dust3r and load the model. Override via
# MAST3R_ROOT / MAST3R_MODEL. Local ckpt absent -> MAST3R_MODEL unset -> HF hub id fallback.
export MAST3R_ROOT="${MAST3R_ROOT:-/root/project/mast3r}"
if [[ -d "$MAST3R_ROOT" ]]; then
  export PYTHONPATH="$MAST3R_ROOT:$MAST3R_ROOT/dust3r:${PYTHONPATH:-}"
  _mm="$MAST3R_ROOT/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
  [[ -z "${MAST3R_MODEL:-}" && -f "$_mm" ]] && export MAST3R_MODEL="$_mm"
fi
# DA3 (Depth Anything 3) estimator import path (src layout); harmless if pip-installed.
export DA3_ROOT="${DA3_ROOT:-/root/project/Depth-Anything-3}"
[[ -d "$DA3_ROOT" ]] && export PYTHONPATH="$DA3_ROOT/src:${PYTHONPATH:-}"

# variant -> filename filter + output suffix. `_rev` sits BEFORE `_nomap` because inference
# builds the name as out_mp4.replace(".mp4", "_nomap.mp4") on sample_XX{_rev}.mp4.
sample_re_for() {
  case "$1" in
    map)   echo 'sample_[0-9]+(_rev)?\.mp4$' ;;
    nomap) echo 'sample_[0-9]+(_rev)?_nomap\.mp4$' ;;
  esac
}
suffix_for() { [[ "$1" == "nomap" ]] && echo "_nomap" || echo ""; }

echo "==== eval on: $RESULTS_DIR   (variant: $VARIANT) ===="
rc=0

for V in "${VARIANTS[@]}"; do
  SRE="$(sample_re_for "$V")"
  SUF="$(suffix_for "$V")"
  nvid=$(ls "$RESULTS_DIR"/sample_*.mp4 2>/dev/null | xargs -n1 basename 2>/dev/null \
         | grep -cE "$SRE" || true)
  echo
  echo "######## variant=$V  ($nvid mp4 matching /$SRE/) ########"
  echo "  camera=$RUN_CAMERA quality=$RUN_QUALITY met3r=$RUN_MET3R"
  if [[ "$nvid" == "0" ]]; then
    echo "[warn] no videos match /$SRE/ -- skipping variant '$V'"
    [[ "$V" == "nomap" ]] && echo "       (a results dir only has _nomap.mp4 if inference ran with --worldmap_tokens)"
    rc=1; continue
  fi

  if [[ "$RUN_CAMERA" == "1" ]]; then
    echo; echo "---- [1/3] camera control  (RotErr / TransErr / CamMC / ADE) ----"
    "$PY" "$CAM" --results_dir "$RESULTS_DIR" \
        --estimators "$ESTIMATORS" \
        --sample_re "$SRE" \
        --out_csv "$RESULTS_DIR/camera_control_metrics${SUF}.csv" \
      || { echo "[warn] camera-control eval failed ($V)"; rc=1; }
  fi

  if [[ "$RUN_QUALITY" == "1" ]]; then
    echo; echo "---- [2/3] image quality  (PSNR / SSIM / LPIPS) ----"
    "$PY" "$QUAL" --results_dir "$RESULTS_DIR" \
        --lpips_net "$LPIPS_NET" \
        --sample_re "$SRE" \
        --out_csv "$RESULTS_DIR/video_quality_metrics${SUF}.csv" \
      || { echo "[warn] image-quality eval failed ($V)"; rc=1; }
  fi

  if [[ "$RUN_MET3R" == "1" ]]; then
    echo; echo "---- [3/3] MEt3R revisitation consistency (palindrome) ----"
    "$PY" "$MET3R" "$RESULTS_DIR" \
        --batch_size "$MET3R_BATCH" \
        --sample_re "$SRE" \
        --out "$RESULTS_DIR/met3r_results${SUF}.json" \
      || { echo "[warn] met3r eval failed ($V)"; rc=1; }
  else
    echo; echo "---- [3/3] MEt3R skipped (set RUN_MET3R=1 for palindrome videos) ----"
  fi
done

# ---------------------------- summary table ----------------------------
RESULTS_DIR="$RESULTS_DIR" EVAL_VARIANTS="${VARIANTS[*]}" "$PY" - <<'PYEOF'
import os, csv, json, math
d = os.environ["RESULTS_DIR"]
variants = os.environ.get("EVAL_VARIANTS", "map").split()
SKIP = {"sample", "error", "n", "dir"}

def csv_means(path):
    if not os.path.exists(path): return None, 0
    rows = list(csv.DictReader(open(path)))
    if not rows: return None, 0
    means = {}
    for k in rows[0].keys():
        if k in SKIP: continue
        vals = []
        for r in rows:
            try:
                f = float(r.get(k, ""))
                if not math.isnan(f): vals.append(f)
            except (TypeError, ValueError):
                pass
        if vals: means[k] = sum(vals) / len(vals)
    return means, len(rows)

def collect(suf):
    """(group, metric) -> mean, for one variant."""
    out, counts = {}, {}
    cam, ncam = csv_means(os.path.join(d, f"camera_control_metrics{suf}.csv"))
    if cam:
        for k, v in cam.items(): out[("camera_control", k)] = v
        counts["camera"] = ncam
    qual, nq = csv_means(os.path.join(d, f"video_quality_metrics{suf}.csv"))
    if qual:
        for k, v in qual.items(): out[("image_quality", k)] = v
        counts["quality"] = nq
    mp = os.path.join(d, f"met3r_results{suf}.json")
    if os.path.exists(mp):
        try:
            s = json.load(open(mp)).get("summary", {})
            for k, v in s.items(): out[("met3r", k)] = float(v)
        except Exception:
            pass
    return out, counts

SUF = {"map": "", "nomap": "_nomap"}
data = {v: collect(SUF[v]) for v in variants}

def better(metric, delta):
    """Verdict on THE MAP, since "does the map help?" is the only question this table exists to
    answer. Saying just "better"/"worse" left the subject open -- it read as a claim about nomap,
    which is the opposite of how anyone reads this table.

    delta is nomap - map, matching the world_map_gain_m / raymap_map_gain convention. That sign
    only means "map helps" for lower-is-better metrics, so the direction has to be resolved per
    metric: errors and LPIPS are lower-better, PSNR/SSIM higher-better, and scale/n/success have
    no direction at all."""
    m = metric.lower()
    if any(t in m for t in ("err", "ade", "fde", "cammc", "rpe", "lpips")):
        lower_is_better = True
    elif any(t in m for t in ("psnr", "ssim")):
        lower_is_better = False
    else:
        return ""
    if abs(delta) < 1e-12: return "same"
    return "map better" if ((delta > 0) == lower_is_better) else "map worse"

print()
print("=" * 74)
print("  SUMMARY  (mean over samples)   " + os.path.basename(os.path.normpath(d)))
for v in variants:
    _, counts = data[v]
    if counts:
        print(f"  [{v}] " + "   ".join(f"{k} n={n}" for k, n in counts.items()))
print("=" * 74)

if len(variants) == 1:
    tbl, _ = data[variants[0]]
    if not tbl:
        print("  (no metric files found)")
    else:
        print(f"  {'group':<15} {'metric':<20} {'mean':>12}")
        print(f"  {'-'*15} {'-'*20} {'-'*12}")
        for (g, k), val in tbl.items():
            print(f"  {g:<15} {k:<20} {val:>12.4f}")
else:
    a, b = variants[0], variants[1]
    ta, tb = data[a][0], data[b][0]
    keys = [k for k in ta if k in tb] or list(ta) or list(tb)
    if not keys:
        print("  (no metric files found)")
    else:
        print(f"  {'group':<15} {'metric':<20} {a:>10} {b:>10} "
              f"{'d(' + b + '-' + a + ')':>13}  verdict")
        print(f"  {'-'*15} {'-'*20} {'-'*10} {'-'*10} {'-'*13}  {'-'*10}")
        for (g, k) in keys:
            va, vb = ta.get((g, k)), tb.get((g, k))
            if va is None or vb is None:
                continue
            dv = vb - va
            print(f"  {g:<15} {k:<20} {va:>10.4f} {vb:>10.4f} {dv:>+13.4f}  {better(k, dv)}")
        print()
        print(f"  verdict is about the MAP: 'map better' = the {a} column wins on that metric.")
        # A map that changes nothing at all is a finding in itself, not a pass.
        if all(abs(tb[k] - ta[k]) < 1e-12 for k in keys):
            print()
            print("  [warn] every metric is bit-identical across variants -- both runs most")
            print("         likely scored the SAME files. Check the --sample_re filters.")
print("=" * 74)
PYEOF

echo; echo "==== done (rc=$rc). outputs in: $RESULTS_DIR ===="
exit $rc

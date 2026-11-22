#!/bin/bash
cd /root/project/evalwam
for D in /root/project/egowam/results_syn/plucker_fidelity/e2/full_all120 /root/project/egowam/results_syn/abl/I_jointfm/ckpt3000/valbench5s_all120 /root/project/egowam/results_syn/goalprog_dl3dv_camfix_d10_dpt_vcons_mode/ckpt500/valbench5s_all120 /root/project/egowam/results_syn/goalprog_mode_real6_nomr_maskfix/ckpt500/valbench5s_all120 /root/project/egowam/results_syn/goalprog_mode_real6_nomr_maskfix/ckpt1000/valbench5s /root/project/egowam/results_syn/goalprog_mode_real6_nomr_maskfix/ckpt2000/valbench5s_all120; do
  [ -f $D/met3r.json ] && continue
  echo "[met3r $(date +%H:%M)] $D"
  CUDA_VISIBLE_DEVICES=${GPU:-3} HF_HOME=/workspace/.hf_home /venv/met3r/bin/python tools/eval_met3r.py --results_dir $D > $D/suite_met3r.log 2>&1 || echo "  failed"
  tail -1 $D/suite_met3r.log
done
python3 tools/report_suite.py > /dev/null
echo MET3R_BACKFILL_DONE

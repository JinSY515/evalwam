#!/bin/bash
# valbench5s extended suite over every run. Ready runs first, then each ablation once its chain is done.
S=/root/project/evalwam/scripts/eval/run_suite.sh; RS=/root/project/egowam/results_syn; B=/root/project/egowam/results_syn/goalprog_mode_real6_nomr_maskfix
$S $RS/plucker_fidelity/e2/full_all120 -
$S $B/ckpt500/valbench5s_all120 -
$S $B/ckpt1000/valbench5s -
$S $B/ckpt2000/valbench5s_all120 -
$S $RS/goalprog_dl3dv_camfix_d10_dpt_vcons_mode/ckpt500/valbench5s_all120 $RS/goalprog_dl3dv_camfix_d10_dpt_vcons_mode/ckpt500/valbench5s
for id in I III IV; do
  until grep -q "ABL_${id}_DONE" $RS/abl/chain_$id.out 2>/dev/null; do sleep 120; done
  case $id in
    I)   P=$RS/abl/I_jointfm/ckpt3000;    HAS_RAYMAP=0 HAS_DEPTH=0 $S $P/valbench5s_all120 $P/valbench5s ;;
    III) P=$RS/abl/III_goalaux/ckpt8000;  HAS_DEPTH=0 CAM_FIX=none $S $P/valbench5s_all120 $P/valbench5s ;;
    IV)  P=$RS/abl/IV_depthpred/ckpt3000; $S $P/valbench5s_all120 $P/valbench5s ;;
  esac
done
echo SUITE_ALL_DONE

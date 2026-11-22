#!/bin/bash
# Suite for the three ablations, each as soon as its (resumed) inference chain prints ABL_<id>_DONE. GPU 2.
S=/root/project/evalwam/scripts/eval/run_suite.sh; RS=/root/project/egowam/results_syn
for id in I III IV; do
  until grep -q "ABL_${id}_DONE" $RS/abl/chain_$id.out 2>/dev/null; do sleep 120; done
  case $id in
    I)   P=$RS/abl/I_jointfm/ckpt3000;    HAS_RAYMAP=0 HAS_DEPTH=0 $S $P/valbench5s_all120 $P/valbench5s ;;
    III) P=$RS/abl/III_goalaux/ckpt8000;  HAS_DEPTH=0 CAM_FIX=none $S $P/valbench5s_all120 $P/valbench5s ;;
    IV)  P=$RS/abl/IV_depthpred/ckpt3000; $S $P/valbench5s_all120 $P/valbench5s ;;
  esac
done
python3 /root/project/evalwam/tools/report_suite.py > /dev/null && echo SUITE_ABL_DONE

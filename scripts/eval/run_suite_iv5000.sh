#!/bin/bash
A=/root/project/egowam/results_syn/abl
O=$A/IV_depthpred/ckpt5000/valbench5s
for d in easy medium hard; do until grep -q "Done\." $A/IV_ck5000_$d.log 2>/dev/null; do grep -q Traceback $A/IV_ck5000_$d.log 2>/dev/null && { echo "TRACEBACK $d"; exit 1; }; sleep 60; done; done
GPU=3 CAM_FIX=opencv /root/project/evalwam/scripts/eval/run_suite.sh $A/IV_depthpred/ckpt5000/valbench5s_all120 $O
python3 /root/project/evalwam/tools/report_suite.py > /dev/null && echo SUITE_IV5000_DONE

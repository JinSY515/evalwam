#!/bin/bash
O=/root/project/egowam/results_syn/goalprog_mode_real6_nomr_maskfix/ckpt3000/valbench5s
for d in easy medium hard; do until grep -q "Done\." $O/$d.log 2>/dev/null; do grep -q Traceback $O/$d.log 2>/dev/null && { echo "TRACEBACK $d"; exit 1; }; sleep 60; done; done
GPU=2 /root/project/evalwam/scripts/eval/run_suite.sh /root/project/egowam/results_syn/goalprog_mode_real6_nomr_maskfix/ckpt3000/valbench5s_all120 $O
python3 /root/project/evalwam/tools/report_suite.py > /dev/null && echo SUITE_CK3000_DONE

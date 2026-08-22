#!/bin/bash
# Relaunch with the jittered line-search optimiser:
#   2 cores  dim-12 calibration (841 from the classical 840 + 1) across seeds
#   1 core   dim-13 Riesz shake-and-relax for 1155 points
#   1 core   dim-14 Fano-design code enumeration (left alone if already up)
cd /home/user/unsolved
# Pin the solver explicitly: riesz.c also implements L-BFGS and the
# published Adam schedule, and this script's numbers come from the
# geometric-homotopy GD that the old committed riesz2 binary ran.
export KISS_SOLVER=gd
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -x opt2; pkill -x riesz2; pkill -f shakeloop.sh; pkill -f rieszloop.sh
sleep 2
: > kissing/logs/calib3.log
nohup sh -c "for s in \$(seq 101 400); do ./kissing/lib/riesz2 12 841 90000 \$s $SP/cl840_841.txt >> kissing/logs/calib3.log 2>&1; done" >/dev/null 2>&1 &
nohup sh -c "for s in \$(seq 501 800); do ./kissing/lib/riesz2 12 841 90000 \$s $SP/cl840_841.txt >> kissing/logs/calib3.log 2>&1; done" >/dev/null 2>&1 &
nohup ./kissing/lib/rieszloop.sh 13 "$SP/ze99_1154.txt" 1155 d13 60000 >/dev/null 2>&1 &
if ! pgrep -f run_fano.sh > /dev/null; then
  nohup ./kissing/lib/run_fano.sh >/dev/null 2>&1 &
fi
sleep 3
ps -eo cmd | grep -cE '[r]iesz2|[f]ano50c'

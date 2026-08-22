#!/bin/bash
# Restart every search with the current binaries: dim-13 shake (penalty
# continuation), dim-13 Riesz continuation, dim-12 calibration (can the same
# machinery reproduce the 841 record from the classical 840?), and the dim-14
# Fano-design code enumeration.
cd /home/user/unsolved
# Pin the solver explicitly: riesz.c also implements L-BFGS and the
# published Adam schedule, and this script's numbers come from the
# geometric-homotopy GD that the old committed riesz2 binary ran.
export KISS_SOLVER=gd
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -x riesz; pkill -x opt2; pkill -f rieszloop.sh; pkill -f shakeloop.sh
pkill -f 'riesz 12'
sleep 2
: > kissing/logs/calib_d12_841.log
: > kissing/logs/calib_d12_842.log
nohup ./kissing/lib/shakeloop.sh 13 "$SP/ze99_1154.txt" 1155 d13 20000 >/dev/null 2>&1 &
nohup ./kissing/lib/rieszloop.sh 13 "$SP/ze99_1154.txt" 1155 d13 30000 >/dev/null 2>&1 &
nohup sh -c "for s in \$(seq 21 60); do ./kissing/lib/riesz 12 841 40000 \$s $SP/cl840_841.txt >> kissing/logs/calib_d12_841.log 2>&1; ./kissing/lib/riesz 12 842 40000 \$s $SP/cl840_842.txt >> kissing/logs/calib_d12_842.log 2>&1; done" >/dev/null 2>&1 &
if ! pgrep -f run_fano.sh > /dev/null; then
  nohup ./kissing/lib/run_fano.sh >/dev/null 2>&1 &
fi
sleep 2
ps -eo pid,cmd | grep -E 'rieszloop|shakeloop|run_fano|riesz 12' | grep -v grep | wc -l

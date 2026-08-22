#!/bin/bash
# Clean-CPU calibration: can the machinery here reproduce the dimension-12
# record (841) from the classical 840?  If it cannot, dimension-13 numerical
# near-misses say nothing about whether 1155 exists.
cd /home/user/unsolved
# Pin the solver explicitly: riesz.c also implements L-BFGS and the
# published Adam schedule, and this script's numbers come from the
# geometric-homotopy GD that the old committed riesz2 binary ran.
export KISS_SOLVER=gd
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -x riesz; pkill -x opt2; pkill -f rieszloop.sh; pkill -f shakeloop.sh
pkill -f run_fano.sh; pkill -f fano50c.py
sleep 2
: > kissing/logs/calib_clean.log
for s in 41 42 43 44; do
  ( ./kissing/lib/riesz 12 841 200000 $s "$SP/cl840_841.txt" >> kissing/logs/calib_clean.log 2>&1 ) &
done
wait
echo done >> kissing/logs/calib_clean.log

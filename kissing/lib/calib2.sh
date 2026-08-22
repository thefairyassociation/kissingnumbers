#!/bin/bash
# Calibrate the line-search Riesz optimiser on the dimension-12 case where a
# solution is known to exist: 840 classical points + 1 -> 841.
cd /home/user/unsolved
# Pin the solver explicitly: riesz.c also implements L-BFGS and the
# published Adam schedule, and this script's numbers come from the
# geometric-homotopy GD that the old committed riesz2 binary ran.
export KISS_SOLVER=gd
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -x riesz
sleep 2
: > kissing/logs/calib2.log
for s in 51 52 53 54; do
  ( ./kissing/lib/riesz2 12 841 120000 $s "$SP/cl840_841.txt" >> kissing/logs/calib2.log 2>&1 ) &
done
wait
echo done >> kissing/logs/calib2.log

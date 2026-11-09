#!/bin/bash
# The numerical dim-13 lane is handed off; stop those streams and leave the
# cores to the dim-12 calibration that still has to finish and to the dim-14
# combinatorial search.
cd /home/user/unsolved
pkill -f rieszloop.sh
pkill -f shakeloop.sh
pkill -x riesz2
sleep 2
ps -eo cmd | grep -cE '[r]iesz_rev|[f]ano50c'

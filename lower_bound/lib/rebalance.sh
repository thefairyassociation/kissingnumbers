#!/bin/bash
# Keep four CPU-bound searches running: dim-13 shake (penalty), dim-13 Riesz
# continuation, dim-14 global-code search, dim-14 Fano-design code enumeration.
cd /home/user/unsolved
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -x mis8
pkill -f 'mis8 14'
sleep 1
if ! pgrep -f rieszloop.sh > /dev/null; then
  nohup ./lower_bound/lib/rieszloop.sh 13 "$SP/ze99_1154.txt" 1155 d13 4000 >/dev/null 2>&1 &
fi
if ! pgrep -f shakeloop.sh > /dev/null; then
  nohup ./lower_bound/lib/shakeloop.sh 13 "$SP/ze99_1154.txt" 1155 d13 8000 >/dev/null 2>&1 &
fi
if ! pgrep -f run_d14.sh > /dev/null; then
  nohup ./lower_bound/lib/run_d14.sh >/dev/null 2>&1 &
fi
if ! pgrep -f run_fano.sh > /dev/null; then
  nohup ./lower_bound/lib/run_fano.sh >/dev/null 2>&1 &
fi
sleep 2
ps -eo pid,cmd | grep -E 'rieszloop|shakeloop|run_d14|run_fano' | grep -v grep | wc -l

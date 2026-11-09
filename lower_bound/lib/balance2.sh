#!/bin/bash
# Keep two calibration runs and put the other two cores back on dimension 13.
cd /home/user/unsolved
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
for p in $(pgrep -f 'riesz2 12 841 120000 53' ; pgrep -f 'riesz2 12 841 120000 54'); do
  kill "$p" 2>/dev/null
done
sleep 2
if ! pgrep -f shakeloop.sh > /dev/null; then
  nohup ./lower_bound/lib/shakeloop.sh 13 "$SP/ze99_1154.txt" 1155 d13 40000 >/dev/null 2>&1 &
fi
if ! pgrep -f run_fano.sh > /dev/null; then
  nohup ./lower_bound/lib/run_fano.sh >/dev/null 2>&1 &
fi
sleep 2
ps -eo cmd | grep -cE '[r]iesz2|[o]pt2|[f]ano50c'

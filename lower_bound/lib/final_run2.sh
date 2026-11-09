#!/bin/bash
# Shorter Riesz runs so restarts actually complete and get logged.
cd /home/user/unsolved
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -x riesz2; pkill -f rieszloop.sh
sleep 2
sed -i 's/riesz2 \$d \$N [0-9]*/riesz2 $d $N 12000/' lower_bound/lib/rieszloop.sh
grep -n 'riesz2 \$d' lower_bound/lib/rieszloop.sh
for t in a b c; do
  nohup ./lower_bound/lib/rieszloop.sh 13 "$SP/ze99_1154.txt" 1155 d13$t $((200000 + RANDOM)) >/dev/null 2>&1 &
  sleep 1
done
if ! pgrep -f run_fano.sh > /dev/null; then
  nohup ./lower_bound/lib/run_fano.sh >/dev/null 2>&1 &
fi
sleep 3
ps -eo cmd | grep -cE '[r]iesz2 13|[f]ano50c'

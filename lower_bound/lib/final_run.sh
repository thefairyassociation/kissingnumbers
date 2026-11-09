#!/bin/bash
# Final allocation: three streams of dim-13 shake-and-relax with the jittered
# line-search Riesz optimiser, one stream of dim-14 Fano-design enumeration.
# Shorter runs so many more restarts get sampled.
cd /home/user/unsolved
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -x riesz2; pkill -f 'riesz2 12'; pkill -f rieszloop.sh
sleep 2
sed -i 's/riesz2 \$d \$N 90000/riesz2 $d $N 30000/' lower_bound/lib/rieszloop.sh
for t in a b c; do
  nohup ./lower_bound/lib/rieszloop.sh 13 "$SP/ze99_1154.txt" 1155 d13$t $((70000 + RANDOM)) >/dev/null 2>&1 &
  sleep 1
done
if ! pgrep -f run_fano.sh > /dev/null; then
  nohup ./lower_bound/lib/run_fano.sh >/dev/null 2>&1 &
fi
sleep 3
ps -eo cmd | grep -cE '[r]iesz2 13|[f]ano50c'

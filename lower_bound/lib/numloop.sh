#!/bin/bash
# usage: numloop.sh <dim> <N> <seedfile> <tag> <startseed>
cd /home/user/unsolved
d=$1; N=$2; sf=$3; tag=$4; s=$5
log=lower_bound/logs/numeric_$tag.log
for i in $(seq 0 500); do
  r=$(./lower_bound/lib/opt $d $N 400000 $((s+i)) $sf 2>/dev/null)
  echo "$(date -u +%FT%TZ) seed=$((s+i)) $r" >> $log
  best=$(echo $r | sed 's/.*best_max_inner=\([0-9.e+-]*\).*/\1/')
  if awk "BEGIN{exit !($best <= 0.5000000001)}"; then
     echo "$(date -u +%FT%TZ) HIT seed=$((s+i)) $r" >> $log
     cp $sf.n$d.N$N.s$((s+i)).out lower_bound/logs/HIT_${tag}_$((s+i)).txt
  fi
done

#!/bin/bash
# Remove k points from a known configuration, add k+1 random ones, then run the
# continuation optimiser.  usage: shakeloop.sh <dim> <base floats> <N> <tag> <seed0>
cd /home/user/unsolved
d=$1; base=$2; N=$3; tag=$4; s0=$5
SP="${KISS_SCRATCH:-${TMPDIR:-/tmp}/kissing-scratch}"
mkdir -p "$SP"
for i in $(seq 0 20000); do
  s=$((s0+i)); k=$(( (RANDOM % 8) + 1 ))
  python3 kissing/lib/shake2.py "$base" $k $s $SP/shake_$tag.txt >/dev/null 2>&1
  rows=$(wc -l < $SP/shake_$tag.txt)
  if [ "$rows" != "$N" ]; then echo "$(date -u +%FT%TZ) BADROWS $rows" >> kissing/logs/shake_$tag.log; continue; fi
  r=$(./kissing/lib/opt2 $d $N 120000 $s $SP/shake_$tag.txt 2>/dev/null)
  best=$(echo "$r" | sed 's/.*best_max_inner=\([0-9.e+-]*\).*/\1/')
  echo "$(date -u +%FT%TZ) k=$k seed=$s $r" >> kissing/logs/shake_$tag.log
  if awk "BEGIN{exit !($best <= 0.50000001)}"; then
     echo "$(date -u +%FT%TZ) HIT k=$k seed=$s $r" >> kissing/logs/shake_$tag.log
     cp $SP/shake_$tag.txt.n$d.N$N.s$s.out kissing/logs/HIT_${tag}_$s.txt 2>/dev/null
  fi
  rm -f $SP/shake_$tag.txt.n$d.N$N.s$s.out
done

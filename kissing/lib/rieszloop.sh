#!/bin/bash
# Riesz-energy continuation from shaken copies of a known configuration.
# usage: rieszloop.sh <dim> <base floats> <N> <tag> <seed0>
cd /home/user/unsolved
# Pin the solver explicitly: riesz.c also implements L-BFGS and the
# published Adam schedule, and this script's numbers come from the
# geometric-homotopy GD that the old committed riesz2 binary ran.
export KISS_SOLVER=gd
d=$1; base=$2; N=$3; tag=$4; s0=$5
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
for i in $(seq 0 20000); do
  s=$((s0+i)); k=$(( (RANDOM % 12) + 1 ))
  python3 kissing/lib/shake2.py "$base" $k $s $SP/rz_$tag.txt >/dev/null 2>&1
  rows=$(wc -l < $SP/rz_$tag.txt)
  if [ "$rows" != "$N" ]; then continue; fi
  r=$(./kissing/lib/riesz2 $d $N 12000 $s $SP/rz_$tag.txt 2>/dev/null)
  best=$(echo "$r" | sed 's/.*best_max_inner=\([0-9.e+-]*\).*/\1/')
  echo "$(date -u +%FT%TZ) k=$k seed=$s $r" >> kissing/logs/riesz_$tag.log
  if awk "BEGIN{exit !($best <= 0.50000001)}"; then
     echo "$(date -u +%FT%TZ) HIT k=$k seed=$s $r" >> kissing/logs/riesz_$tag.log
     cp $SP/rz_$tag.txt.riesz.s$s.out kissing/logs/HIT_riesz_${tag}_$s.txt 2>/dev/null
  fi
  rm -f $SP/rz_$tag.txt.riesz.s$s.out
done

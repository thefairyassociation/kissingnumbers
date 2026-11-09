#!/bin/bash
# The BLAS/L-BFGS rewrite is ~18x faster.  The branch spends that on MORE, SHORT
# restarts (~15 s each) and lands at 0.51123 -- worse than the single long
# gradient-descent run that reached 0.50519.  This tests the other way of
# spending it: ONE much finer continuation (smaller KISS_SMUL = more exponent
# stages, more steps at each), which is what the homotopy actually wants.
#   usage: spend_the_speedup.sh <riesz binary> <seconds>
cd /home/user/unsolved
BIN=$1
BUDGET=${2:-900}
SP="${KISS_SCRATCH:-/tmp/kissing-scratch}"
LIB=$(python3 -c "import scipy_openblas32 as s; print(s.get_lib_dir())")
SEED=$SP/cl840_841.txt
OUT=kissing/logs/spend_speedup.log
: > $OUT
for cfg in "1.04 400000" "1.02 800000" "1.12 400000"; do
  set -- $cfg
  smul=$1; steps=$2
  s=$(( RANDOM + 7000 ))
  start=$(date +%s)
  r=$(LD_LIBRARY_PATH=$LIB KISS_JIT=0.02 KISS_SMUL=$smul KISS_INNER=16 \
      OMP_NUM_THREADS=4 timeout $BUDGET "$BIN" 12 841 $steps $s "$SEED" 2>/dev/null \
      | grep -o 'best_max_inner=[0-9.e+-]*' | sed 's/.*=//')
  echo "KISS_SMUL=$smul steps=$steps  ->  ${r:-timeout}  ($(( $(date +%s)-start ))s)" | tee -a $OUT
done
echo "baseline to beat: 0.50519 (single long gradient-descent run); branch reports 0.51123" | tee -a $OUT

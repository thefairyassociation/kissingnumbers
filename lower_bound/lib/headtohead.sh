#!/bin/bash
# Equal-wall-clock comparison of the two Riesz optimisers on the dimension-12
# calibration (classical 840 + 1 -> 841, where a solution provably exists).
# Each side gets the same number of seconds and runs as many jittered restarts
# as it can fit; we report the best max inner product each reached.
#   usage: headtohead.sh <seconds-per-side> <path-to-riesz_rev> <path-to-riesz_mine>
cd /home/user/unsolved
BUDGET=${1:-300}
REV=$2
MINE=$3
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
LIB=$(python3 -c "import scipy_openblas32 as s; print(s.get_lib_dir())")
SEED=$SP/cl840_841.txt

run_side () {                       # $1 label  $2 command prefix  $3 steps
  local label=$1 pre=$2 steps=$3
  local end=$(( $(date +%s) + BUDGET )) n=0 best=9
  while [ $(date +%s) -lt $end ]; do
    n=$((n+1))
    r=$(eval "$pre 12 841 $steps $((1000+n)) $SEED" 2>/dev/null | grep -o 'best_max_inner=[0-9.e+-]*' | sed 's/.*=//')
    [ -z "$r" ] && continue
    best=$(python3 -c "print(min($best,$r))")
  done
  echo "$label: $n restarts in ${BUDGET}s, best max inner product $best"
}

run_side "BLAS+L-BFGS (4 threads)" "LD_LIBRARY_PATH=$LIB KISS_JIT=0.03 KISS_INNER=16 OMP_NUM_THREADS=4 $REV" 80000
run_side "hand loops (1 core)     " "KISS_JIT=0.03 $MINE" 80000

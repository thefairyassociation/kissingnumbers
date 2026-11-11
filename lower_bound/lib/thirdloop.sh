#!/bin/bash
cd /home/user/unsolved
for i in $(seq 0 400); do
  r=$(KISS_T=0.33333333333333331 ./lower_bound/lib/opt 12 169 300000 $((3000+i)) "$1" 2>/dev/null)
  echo "$(date -u +%FT%TZ) seed=$((3000+i)) $r" >> lower_bound/logs/third12.log
done

#!/bin/bash
# Long-running enumeration of global codes V for the dim-14 Fano-product design.
cd /home/user/unsolved
for s in $(seq 300 400); do
  python3 lower_bound/dim14/fano50c.py $s 600 >> lower_bound/logs/d14_fano.log 2>&1
done

#!/bin/bash
# Long-running dim-14 global-code search over [14,5,6] codes.
cd /home/user/unsolved
for s in $(seq 201 260); do
  python3 lower_bound/dim14/search_v.py $s 700 >> lower_bound/logs/d14_gcode.log 2>&1
done

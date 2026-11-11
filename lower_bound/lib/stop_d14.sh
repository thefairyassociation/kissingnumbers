#!/bin/bash
# The generic dim-14 random-code search plateaus at ~13 supports; stop it and
# leave the CPU to the Fano-design enumeration and the dim-13 searches.
pkill -f run_d14.sh
sleep 1
pkill -f search_v.py
sleep 1
echo stopped

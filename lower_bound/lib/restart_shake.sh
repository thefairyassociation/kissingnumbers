#!/bin/bash
# Restart the dim-13 shake loop cleanly (kills any previous loop and optimiser).
cd /home/user/unsolved
SP=/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad
pkill -f shakeloop.sh
pkill -x opt2
sleep 2
nohup ./lower_bound/lib/shakeloop.sh 13 "$SP/ze99_1154.txt" 1155 d13 8000 >/dev/null 2>&1 &
sleep 1
echo "restarted"

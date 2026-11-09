#!/bin/bash
# Session briefing: what changed since this branch's last commit.
#
# Printed on stdout by the SessionStart hook, which puts it into the model's
# context at the start of the session, so a new session opens already knowing
# what other branches have landed and whether any search actually found
# something.  Safe to run by hand at any time.
cd "${CLAUDE_PROJECT_DIR:-/home/user/unsolved}" 2>/dev/null || exit 0
MINE=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0

git fetch --all --prune --quiet 2>/dev/null

echo "=== kissing-number briefing ==="
echo "branch $MINE @ $(git log -1 --format=%h\ %s 2>/dev/null | cut -c1-72)"

# 1. Anything a search actually found outranks everything else here.
hits=$(ls kissing/logs/HIT_*.txt 2>/dev/null | wc -l)
if [ "$hits" -gt 0 ]; then
  echo "!! $hits HIT file(s) in kissing/logs/ -- a search reported a feasible"
  echo "!! configuration. VERIFY IT before believing it: recompute the Gram from"
  echo "!! the file in numpy, then run kissing/lib/verify_exact.py. Two false"
  echo "!! 'feasible' results have been produced by that code before."
else
  echo "searches: no HIT files (nothing claimed feasible)"
fi

# 2. Branches carrying work that has NOT been reviewed yet.  kissing/.reviewed
#    records "branch sha" for every tip already looked at, so branches whose
#    content was imported long ago stop showing up as noise.
REV=kissing/.reviewed
touch "$REV" 2>/dev/null
echo "UNREVIEWED branches:"
found=0
for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin 2>/dev/null); do
  [ "$b" = "origin/HEAD" ] && continue
  [ "$b" = "origin/$MINE" ] && continue
  tip=$(git rev-parse --short "$b" 2>/dev/null) || continue
  grep -q "^$b $tip$" "$REV" 2>/dev/null && continue
  cnt=$(git rev-list --count "$MINE..$b" 2>/dev/null) || continue
  [ "${cnt:-0}" -eq 0 ] && continue
  found=1
  echo "  $b @ $tip: $cnt commit(s) not in this branch"
  git log --format='    - %s' -3 "$MINE..$b" 2>/dev/null | cut -c1-88
done
[ "$found" -eq 0 ] && echo "  (none -- everything on the remote has been reviewed)"

# 3. Standing state, so the session does not have to re-derive it.
echo "records still standing: dim12 841, dim13 1154, dim14 1932; best verified"
echo "  here 840 / 1154 / 1932, all reproductions (kissing/best.json)"
echo "dim-12 N=841 calibration to beat: 0.50519  (target < 0.5; published 0.4999999)"
echo "read kissing/README.md for state, kissing/HANDOFF_optimizer.md for the open task."
if [ "$found" -eq 1 ]; then
  echo "NOTE: review any new numbers against kissing/REVIEW_optimizer_branch.md"
  echo "  before trusting them -- check the build actually rebuilt, that no"
  echo "  -ffast-math crept into riesz.c, and recompute maxima from the .out files."
fi
echo "=== end briefing ==="

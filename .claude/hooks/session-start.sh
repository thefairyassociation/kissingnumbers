#!/bin/bash
# SessionStart hook: bring a cold container back to a working state and resume
# the long-running kissing-number searches.
#
# The container is reclaimed whenever the chat is closed, and comes back with no
# Python packages installed and nothing running.  Each search round is
# independent (shake -> optimise -> append to log), so nothing is lost across a
# restart except the round that was in flight; this hook just gets the machine
# working again and puts the cores back to work.
set -euo pipefail

# Only in the remote (web) environment; a local checkout should stay quiet.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-/home/user/unsolved}"

# 1. Python dependencies used by the verifier, the LP hole search and the
#    structural searches.  Idempotent; the container image caches these.
python3 - <<'PY' || pip install --quiet numpy scipy sympy
import numpy, scipy, sympy  # noqa: F401
PY

# 2. C tools.  Rebuilt only when missing, so a warm container skips this.
for t in mis8 clique gclique addable signmis; do
  [ -x "kissing/lib/$t" ] || gcc -O2 -o "kissing/lib/$t" "kissing/lib/$t.c" -lm
done
[ -x kissing/lib/opt2 ] || { gcc -O3 -march=native -ffast-math -o kissing/lib/opt kissing/lib/opt.c -lm && cp -f kissing/lib/opt kissing/lib/opt2; }
# riesz has no -ffast-math on purpose: it would compile away the NaN guard that
# stops a pow() overflow from being reported as a feasible configuration.
[ -x kissing/lib/riesz2 ] || gcc -O3 -march=native -o kissing/lib/riesz2 kissing/lib/riesz.c -lm

# 3. Regenerate the float seeds the searches start from (scratch is not persisted).
SP="${TMPDIR:-/tmp}/kissing-scratch"
mkdir -p "$SP"
[ -s "$SP/ze99_1154.txt" ] || python3 kissing/lib/tofloat.py \
  kissing/dim13/configs/ze99_1154_exact.json "$SP/ze99_1154.txt" 0 >/dev/null

# 4. Resume the searches, unless already running or KISSING_NO_RESUME is set.
if [ "${KISSING_NO_RESUME:-}" != "1" ] && ! pgrep -f rieszloop.sh >/dev/null 2>&1; then
  mkdir -p kissing/logs
  for tag in a b c; do
    KISS_SCRATCH="$SP" nohup ./kissing/lib/rieszloop.sh 13 "$SP/ze99_1154.txt" 1155 "d13$tag" \
      $((RANDOM + 300000)) >/dev/null 2>&1 &
  done
  nohup ./kissing/lib/run_fano.sh >/dev/null 2>&1 &
  echo "resumed 3 dim-13 search streams + the dim-14 enumeration" >&2
fi

# 5. Briefing on stdout.  SessionStart hook stdout is added to the session
#    context, so the next session opens already knowing what landed on other
#    branches and whether any search claimed a feasible configuration.
./kissing/lib/briefing.sh 2>/dev/null || true

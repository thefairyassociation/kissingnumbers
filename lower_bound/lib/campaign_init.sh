#!/usr/bin/env bash
# Write the starting configuration for a search-campaign arm.  Run from the
# repository root:   lower_bound/lib/campaign_init.sh KIND OUTFILE
#
#   d12-841   classical 840 (Clebsch / Leech-Sloane) + one hypercube vertex, 841 rows
#   d12-840   the same 840 alone (use with campaign.py --extra hypercube|gauss)
#   d13-1154  Zinoviev-Ericson 1154 alone (use with --extra hypercube|gauss)
#   d14-1932  Ganzhinov 1932 alone (use with --extra hypercube|gauss)
set -euo pipefail
kind="$1"; out="$2"
case "$kind" in
  d12-841)  python3 lower_bound/lib/seed841.py "$out" --mode hypercube ;;
  d12-840)  python3 lower_bound/lib/seed841.py "$out.tmp" --mode hypercube
            head -n 840 "$out.tmp" > "$out"; rm -f "$out.tmp" ;;
  d13-1154) python3 lower_bound/lib/tofloat.py lower_bound/dim13/configs/ze99_1154_exact.json "$out" 0 ;;
  d14-1932) python3 - "$out" <<'PY'
import json, sys, math
d = json.load(open("lower_bound/dim14/configs/ganzhinov_1932_exact.json"))
V = [[int(c) for c in v] for v in d["vectors"]]
with open(sys.argv[1], "w") as f:
    for v in V:
        s = math.sqrt(sum(c * c for c in v))
        f.write(" ".join("%.17g" % (c / s) for c in v) + "\n")
print(f"wrote {len(V)} rows")
PY
  ;;
  *) echo "unknown init kind: $kind" >&2; exit 2 ;;
esac
wc -l "$out"

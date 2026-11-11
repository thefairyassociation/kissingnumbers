# Search campaign `c01-smoke`

Workflow run: https://github.com/thefairyassociation/kissingnumbers/actions/runs/42698953799

Written by `lower_bound/lib/campaign_collect.py`; the numbers are recomputed from the
coordinate files, and a hit is listed only if `certify_float.py` passed it (exact integer
arithmetic).

| arm | n | N | shards | CPU-hours | seeds screened | roots | best max inner product | certified below 1/2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d12-file | 12 | 841 | 1 | 0.4 | 40 | 1 | 0.5002991423 | no |
| d14-cube | 14 | 1933 | 1 | 1.4 | 40 | 0 | - | no |

Near-antipodal pairs after the first exponent stage (bucketed by 50), per arm:

* `d12-file`: 50+: 2, 100+: 24, 150+: 12, 200+: 1, 400+: 1
* `d14-cube`: 600+: 40

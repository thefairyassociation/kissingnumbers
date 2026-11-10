# Search campaign `c04-beyond`

Workflow run: https://github.com/thefairyassociation/kissingnumbers/actions/runs/42785278093

Written by `lower_bound/lib/campaign_collect.py`; the numbers are recomputed from the
coordinate files, and a hit is listed only if `certify_float.py` passed it (exact integer
arithmetic).

| arm | n | N | shards | CPU-hours | seeds screened | roots | best max inner product | certified below 1/2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d12-842 | 12 | 842 | 6 | 119.8 | 1760 | 1680 | 0.5009716487 | no |
| d13-s16 | 13 | 1155 | 10 | 200.1 | 1320 | 1296 | 0.5060089561 | no |
| d13-s32 | 13 | 1155 | 4 | 80.3 | 504 | 492 | 0.5063593108 | no |

Near-antipodal pairs after the first exponent stage (bucketed by 50), per arm:

* `d12-842`: 400+: 1760
* `d13-s16`: 350+: 1, 400+: 395, 450+: 919, 500+: 5
* `d13-s32`: 550+: 504

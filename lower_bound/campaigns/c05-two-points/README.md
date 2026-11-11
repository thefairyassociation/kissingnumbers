# Search campaign `c05-two-points`

Workflow run: https://github.com/thefairyassociation/kissingnumbers/actions/runs/42894869349

Written by `lower_bound/lib/campaign_collect.py`; the numbers are recomputed from the
coordinate files, and a hit is listed only if `certify_float.py` passed it (exact integer
arithmetic).

| arm | n | N | shards | CPU-hours | seeds screened | roots | best max inner product | certified below 1/2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d12-842-melt | 12 | 842 | 20 | 399.2 | 61320 | 2341 | 0.5009653797 | no |

Near-antipodal pairs at the screening point (bucketed by 50), per arm. The screen is applied
after the first exponent stage a seed runs, which depends on the arm's entry stage:

* `d12-842-melt` (after the s = 8 stage): 50+: 207, 100+: 40101, 150+: 17415, 200+: 1141, 250+: 103, 300+: 52, 350+: 290, 400+: 2011

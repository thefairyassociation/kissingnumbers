# Search campaign `c02-scale`

Workflow run: https://github.com/thefairyassociation/kissingnumbers/actions/runs/42700332843

Written by `lower_bound/lib/campaign_collect.py`; the numbers are recomputed from the
coordinate files, and a hit is listed only if `certify_float.py` passed it (exact integer
arithmetic).

| arm | n | N | shards | CPU-hours | seeds screened | roots | best max inner product | certified below 1/2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d12-cube | 12 | 841 | 4 | 79.8 | 10480 | 358 | 0.5000060565 | no |
| d12-file | 12 | 841 | 11 | 220.0 | 28440 | 1200 | 0.5000897150 | no |
| d14-cube | 14 | 1933 | 2 | 40.7 | 304 | 296 | 0.5189808810 | no |

Near-antipodal pairs after the first exponent stage (bucketed by 50), per arm:

* `d12-cube`: 50+: 62, 100+: 6875, 150+: 2930, 200+: 227, 250+: 27, 300+: 10, 350+: 39, 400+: 310
* `d12-file`: 50+: 134, 100+: 17391, 150+: 8958, 200+: 708, 250+: 45, 300+: 20, 350+: 124, 400+: 1060
* `d14-cube`: 600+: 304

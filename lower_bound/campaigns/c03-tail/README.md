# Search campaign `c03-tail`

Workflow run: https://github.com/thefairyassociation/kissingnumbers/actions/runs/42746402869

Written by `lower_bound/lib/campaign_collect.py`; the numbers are recomputed from the
coordinate files, and a hit is listed only if `certify_float.py` passed it (exact integer
arithmetic).

| arm | n | N | shards | CPU-hours | seeds screened | roots | best max inner product | certified below 1/2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d12-cube | 12 | 841 | 16 | 319.6 | 48160 | 1685 | 0.4999967975 | **YES** (2) |
| d13-cube | 13 | 1155 | 4 | 80.1 | 13200 | 0 | - | no |

Near-antipodal pairs after the first exponent stage (bucketed by 50), per arm:

* `d12-cube`: 50+: 244, 100+: 31741, 150+: 13437, 200+: 943, 250+: 99, 300+: 48, 350+: 202, 400+: 1446
* `d13-cube`: 50+: 12956, 100+: 244

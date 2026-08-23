# Exact ZE99 Steiner-switch augmentation search

Generated `2026-08-23T20:03:50.406278+00:00`.

## Result

- Current exact record: **1154**.
- Candidate count if successful: **1154**.
- Beats the record: **False**.
- Complete norm-16 integer family exhausted: **True**.
- Integer candidates tested: **8192**.

The fixed ZE99 tetrad, axial, and irrational layers reduce every compatible
norm-16 integer vector to the 8,192-vector diamond shell. The program
reconstructs all one-conflict diamond replacements, then exhaustively tests
whether each shell vector can be added after mutually compatible zero-cost
repairs of all its diamond conflicts.

A positive result is emitted only after the full 1,155-vector configuration
passes the exact Q(sqrt(3)) verifier.

## Switch catalogue

```json
{
  "baseline_diamond_count": 288,
  "baseline_shell_rows": 288,
  "conflict_count_distribution_over_shell": {
    "1": 288,
    "14": 240,
    "15": 640,
    "16": 768,
    "17": 1920,
    "18": 480,
    "2": 240,
    "4": 960,
    "5": 960,
    "6": 1696
  },
  "matches_reported_528_to_264_and_24_skeleton": false,
  "nonbaseline_shell_rows": 7904,
  "one_conflict_replacement_candidates": 0,
  "replacement_group_size_distribution": {},
  "replacement_pair_conflicts_across_groups": 0,
  "replacement_pair_conflicts_within_same_group": 0,
  "rigid_diamond_indices": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    285,
    286,
    287
  ],
  "rigid_directed_diamonds": 288,
  "shell_size": 8192,
  "touchable_directed_diamonds": 0
}
```

## Search summary

```json
{
  "baseline_duplicates": 288,
  "candidate_conflict_count_distribution": {
    "1": 288,
    "14": 240,
    "15": 640,
    "16": 768,
    "17": 1920,
    "18": 480,
    "2": 240,
    "4": 960,
    "5": 960,
    "6": 1696
  },
  "direct_extensions_without_repairs": 0,
  "near_misses": [
    {
      "baseline_diamond_conflicts": [
        21
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        21
      ],
      "vector": [
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        99
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        99
      ],
      "vector": [
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        191
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        191
      ],
      "vector": [
        1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        51
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        51
      ],
      "vector": [
        -1,
        -1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        137
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        137
      ],
      "vector": [
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        87
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        87
      ],
      "vector": [
        -1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        159
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        159
      ],
      "vector": [
        1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        63
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        63
      ],
      "vector": [
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        9
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        9
      ],
      "vector": [
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        69
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        69
      ],
      "vector": [
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        241
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        241
      ],
      "vector": [
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        153
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        153
      ],
      "vector": [
        1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        35
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        35
      ],
      "vector": [
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        173
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        173
      ],
      "vector": [
        1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        77
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        77
      ],
      "vector": [
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        41
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        41
      ],
      "vector": [
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        269
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        269
      ],
      "vector": [
        1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        265
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        265
      ],
      "vector": [
        -1,
        1,
        1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        11
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        11
      ],
      "vector": [
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        205
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        205
      ],
      "vector": [
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        139
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        139
      ],
      "vector": [
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        273
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        273
      ],
      "vector": [
        1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        27
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        27
      ],
      "vector": [
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        133
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        133
      ],
      "vector": [
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        281
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        281
      ],
      "vector": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        65
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        65
      ],
      "vector": [
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        107
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        107
      ],
      "vector": [
        -1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        25
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        25
      ],
      "vector": [
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        3
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        3
      ],
      "vector": [
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        1,
        -2
      ]
    },
    {
      "baseline_diamond_conflicts": [
        5
      ],
      "baseline_duplicate": true,
      "conflict_count": 1,
      "empty_repair_domain_count": 0,
      "empty_repair_domains": [],
      "rigid_conflict_count": 1,
      "rigid_conflicts": [
        5
      ],
      "vector": [
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -2
      ]
    }
  ],
  "nonbaseline_tested": 7904,
  "rejected_by_empty_repair_domain": 0,
  "rejected_by_replacement_pair_conflicts": 0,
  "rejected_by_rigid_diamond": 8192,
  "repairable_candidates": 0,
  "tested": 8192,
  "total_backtracking_nodes": 0
}
```

# Kissing number in dimension 12

Autonomous search for a **kissing arrangement** in \(\mathbb{R}^{12}\): unit vectors
with all pairwise inner products \(\le 1/2\), **strictly more than the live record**,
certified in **exact arithmetic**. Floats are used only for search.

## Live record (fetched 2026-08-21)

Source: [Henry Cohn, Kissing numbers](https://cohn.mit.edu/kissing-numbers)

| dim | lower | upper | ratio | refs |
| --- | --- | --- | --- | --- |
| 12 | **841** | **1355** | 1.612 | [24], [10] |

- Lower bound 841: Takhanov, Assylbekov, Yun 2026, [arXiv:2606.18984](https://arxiv.org/abs/2606.18984)
- Upper bound 1355: de Laat–Leijenhorst 2024, [doi:10.1007/s12532-024-00264-w](https://doi.org/10.1007/s12532-024-00264-w)

**Reproducing 841 is a baseline, not progress.** Success requires a verified count \(> 841\).

The 841 configuration is numerical (max inner product \(0.499999937751\)) obtained by
logarithmic Riesz-energy continuation from the classical 840-point arrangement plus
one extra \(({\pm1})^{12}/\sqrt{12}\) seed. The authors also report unpolished 842-
and 844-point spherical codes with max inner products \(0.50090\) and \(0.50440\)
(above \(1/2\)).

## Classical constructions (this directory)

| count | family | exact? | notes |
| --- | --- | --- | --- |
| 756 | Coxeter–Todd \(K_{12}\) minima | yes, \(\mathbb{Q}(\sqrt{3})\) | lattice kissing number |
| 756 | laminated \(\Lambda_{12}\) | lattice | same kissing number as \(K_{12}\), different lattice |
| 840 | Leech–Sloane / Clebsch \(48\)-system + \(K_6\) bridges | yes, \(\mathbb{Q}(\sqrt{2})\) | two \(60\)-point blocks in complementary \(\mathbb{R}^6\) plus 720 bridges |
| 841 | Takhanov–Assylbekov–Yun | numerical only | not claimed here as an exact certificate |

### 840 structure (Leech–Sloane 1971, flexibility: arXiv:2606.18984)

- \(60\) vectors in \(\mathbb{R}^6\times\{0\}\) and \(60\) in \(\{0\}\times\mathbb{R}^6\)
- Each \(60\)-block = \(12\) signed coordinate vectors \({\pm e_i}\) + a **\(48\)-system**
- Example 1 \(48\)-system: Clebsch equator in \(\mathbb{R}^5\) with
  \(x_\varepsilon=(\sqrt{2}/3\,\varepsilon_1,\ldots,\sqrt{2}/3\,\varepsilon_4,1/3\,\varepsilon_5)\)
  and analytic floors \(b_\varepsilon\), plus the \(O(4)\) family of Theorem 2
- \(720\) bridges from the unique \(1\)-factorization of \(K_6\): same-color edges
  \(\{i,j\}\) and \(\{k,\ell\}\), vectors \(\frac12(\varepsilon_i e_i+\varepsilon_j e_j+\varepsilon_k e_{6+k}+\varepsilon_\ell e_{6+\ell})\)

### Coxeter–Todd \(K_{12}\)

Eisenstein lattice \(A_6^{(2)}\) via Construction A on the hexacode
([Conway–Sloane 1983](https://doi.org/10.1017/S0305004100060746),
[Nebe–Sloane catalogue](https://www.math.rwth-aachen.de/~Gabriele.Nebe/LATTICES/K_{12}.html)
— file `K12.html`). Real form has \(756\) minima of norm \(4\).

Related: Nordstrom–Robinson / hexacode, Mitchell group \(6.\mathrm{PSU}_4(\mathbb{F}_3).2\).

## What cannot be a record

- \(K_{12}\) minima alone (\(756 < 841\))
- Rigid \(840\) plus a coordinate-hypercube vector \((\pm1)^{12}/\sqrt{12}\): every used
  \(4\)-support of a bridge appears with **all** \(16\) sign patterns, so some bridge
  meets that extra vector in inner product \(1/\sqrt{3}>1/2\). Adding such a point
  **requires moving the bridges** (as in the 2026 numerical search).

## Files

- `verify.py` — exact verifier (sympy / \(\mathbb{Q}(\sqrt{2})\))
- `constructions/clebsch840.py` — exact \(840\)
- `constructions/k12.py` — exact \(K_{12}\) minima
- `constructions/leech.py` — Golay \(G_{24}\) and Leech sections
- `constructions/codes.py` — ternary Golay spherical vectors
- `search/` — algebraic extras, hole search, \(4{+}4{+}4\) greedy, Riesz continuation
- `configs/` — verified configurations (exact coordinates as algebraic strings)
- `best.json` — best **exactly verified** count so far
- `progress.log` — append-only log
- `verifier_output.txt` — verifier stdout for the best attempt / success

## Citations (URLs)

- Cohn table: https://cohn.mit.edu/kissing-numbers
- Takhanov–Assylbekov–Yun 2026: https://arxiv.org/abs/2606.18984
- Leech–Sloane 1971: https://doi.org/10.4153/CJM-1971-081-3
- Ganzhinov 2025 (dims 10/11/14, method of highly symmetric lines): https://doi.org/10.1016/j.laa.2025.05.002
- Conway–Sloane \(K_{12}\): https://doi.org/10.1017/S0305004100060746
- Nebe–Sloane \(K_{12}\): https://www.math.rwth-aachen.de/~Gabriele.Nebe/LATTICES/K12.html
- de Laat–Leijenhorst upper bounds: https://doi.org/10.1007/s12532-024-00264-w
- Classification of 840-point arrangements: Takhanov–Yun, https://arxiv.org/abs/2606.03299

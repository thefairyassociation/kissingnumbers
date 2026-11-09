# Kissing number in dimension 14

## Live record used

Fetched 2026-08-21 from [Henry Cohn, Kissing numbers](https://cohn.mit.edu/kissing-numbers):

| Dimension | Lower bound | Upper bound | Ratio | References |
| --- | --- | --- | --- | --- |
| 14 | **1932** | **3174** | 1.643 | [7, 10] |

- Lower bound: Mikhail Ganzhinov, *Highly symmetric lines*, Linear Algebra Appl. **722** (2025), 12–37, [doi:10.1016/j.laa.2025.05.002](https://doi.org/10.1016/j.laa.2025.05.002), preprint [arXiv:2207.08266](https://arxiv.org/abs/2207.08266).
- Upper bound: D. de Laat and N. Leijenhorst, *Solving clustered low-rank semidefinite programs arising from polynomial optimization*, Math. Program. Comput. **16** (2024), 503–534, [doi:10.1007/s12532-024-00264-w](https://doi.org/10.1007/s12532-024-00264-w).
- Explicit integer coordinates for a 1932-point configuration are in Cohn’s archive [hdl.handle.net/1721.1/153312](https://hdl.handle.net/1721.1/153312) (`dimensions1-24.txt`).

Reproducing 1932 is **baseline, not progress**. Success in this directory is an explicit set of **strictly more than 1932** unit vectors in \(\mathbb{R}^{14}\) with all pairwise inner products \(\le 1/2\), verified in **exact** arithmetic.

## Ganzhinov construction (baseline)

Section 5.5 of [arXiv:2207.08266](https://arxiv.org/abs/2207.08266). Work in \(\mathbb{C}^7 \cong \mathbb{R}^{14}\). Let \(G \cong U(3,3)\) act irreducibly on \(\mathbb{R}^7\). Three highly symmetric frames:

- \(\Phi_1\): 56 vectors, inner shell of \(E_7^*\);
- \(\Phi_2\): 126 vectors, inner shell of \(E_7\), partitioned into **9 cross polytopes** (coordinate frames);
- \(\Phi_3\): 1512 vectors of the form \(\frac{i^k}{\sqrt{2}}(v+iw)\) with \(v\neq w\) on the same frame.

Adding phased copies of \(\Phi_2\) and the \(D_7\) roots \(\Phi_4\) gives the \(\{0,\pm 1/4,\pm 1/2,-1\}\)-angular code

\[
\Phi_3 \cup \tfrac{1+i}{\sqrt{2}}\Phi_2 \cup \tfrac{1-i}{\sqrt{2}}\Phi_2 \cup \Phi_4 \cup i\Phi_4
\]

of cardinality \(1512+126+126+84+84=1932\).

**Integer model** (scale \(2\sqrt{2}\)): vectors in \(\mathbb{Z}^{14}\) of squared length 8. Pairwise inner products \(\le 4\) iff unit cosines \(\le 1/2\). This configuration is exactly:

- all **364** \(D_{14}\) roots of type \((\pm 2,\pm 2,0^{12})\);
- **1568** weight-8 vectors of type \((\pm 1^8,0^6)\).

The 364 type-(2,2) vectors are mutually compatible with *every* weight-8 vector of squared length 8. Beating 1932 in this shell is therefore equivalent to finding an antipodal subset of the 768768 weight-8 vectors with pairwise \(|\langle x,y\rangle|\le 4\) of size **> 1568**.

Implemented in `constructions/e7_frames.py` and `constructions/ganzhinov.py`. Exact Gram check: `python3 lower_bound/dim14/verifier.py --ganzhinov`.

## Search families

| ID | Family | Status |
| --- | --- | --- |
| A | Extra weight-8 vectors compatible with the frozen 1932 | see `progress.log` |
| B | Remove/reinsert swaps on the weight-8 graph | see `progress.log` |
| C | Random greedy independent sets in the weight-8 graph | see `progress.log` |
| D | Type \((2,1^4,0^9)\) trades against type-(2,2) | see `progress.log` |
| E | Cohn–Li all-equal / signed-constant vector in \(\mathbb{Q}(\sqrt{7})\) | see `progress.log` |
| F | Barnes–Wall \(\Lambda_{16}\) coordinate 14-section | see `progress.log` |
| G | Shortened Witt \(S(5,8,24)\) even-sign (Leech-style) | see `progress.log` |
| H | Numerical holes of the 1932 configuration | see `progress.log` |

Related literature that did **not** improve 1932:

- PackingStar (Ma et al., [arXiv:2511.13391](https://arxiv.org/abs/2511.13391)): thousands of 14D configurations, none above 1932; they do improve a *generalized* kissing number \(K(14,1/3)=252\).
- Cohn–Li ([arXiv:2411.04916](https://arxiv.org/abs/2411.04916)): sign-modification of Leech sections, used in dimensions 17–21, not 14.
- Laminated lattice \(\Lambda_{14}\): kissing number 1422 ([Nebe–Sloane catalogue](https://www.math.rwth-aachen.de/homes/Gabriele.Nebe/LATTICES/LAMBDA14.html)).
- Zinoviev–Ericson 1999: previous lower bound 1606, superseded by Ganzhinov.

Constant-weight note: \(A(14,8,8)\le 6\) by the Johnson pair packing (complements are 6-sets intersecting in at most 2 points, \( \binom{14}{2}/\binom{6}{2}=6.06\)). So at most 6 octads with pairwise intersection \(\le 4\), giving at most \(364+128\cdot 6=1132\) vectors if signs are an unrestricted even-weight code per octad. Ganzhinov’s 1568 therefore **requires** 8-supports with intersection 6 and correlated signs.

## Files

- `verifier.py` — exact integer Gram verifier (sympy path available for \(\mathbb{Q}\)-coords).
- `search_extend.py` — structured searches A–H.
- `configs/` — explicit coordinates (`ganzhinov_1932.npy/txt`).
- `best.json` — current best *verified* count.
- `progress.log` — one-liner search log.
- `verifier_output.txt` — verifier stdout.

## How to verify

```bash
python3 lower_bound/dim14/verifier.py --ganzhinov
python3 lower_bound/dim14/verifier.py lower_bound/dim14/configs/ganzhinov_1932.txt
```

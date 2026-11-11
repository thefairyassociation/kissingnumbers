# Where the old branches and pull requests went

Until 2026-11-09 the work in this repository was spread over 17 side branches and
7 open pull requests, written by several different agents between August and
October 2026, most of it never merged. `main` held only the first two days of
work. Everything has now been merged into one tree (`lower_bound/`), and the
commit history of every branch is preserved in that merge.

This page records what each branch contained and what happened to it, so the
branches and pull requests can be closed without anyone having to re-read
them. **Nothing was deleted from GitHub**: the branches and PRs are still
there, and closing them is the repository owner's call.

## Summary

| branch | PR | what it was | status |
| --- | --- | --- | --- |
| `claude/focused-carson-niruot` | – | 2026-10-02: fast Riesz engine, branching search, exact float certifier, `STATUS.md` | merged in full; it was the newest state of the lower-bound work |
| `claude/pr-review-0y5lrk` | [#4](https://github.com/thefairyassociation/kissingnumbers/pull/4) | CI for the self-checks | merged; workflow renamed `checks.yml` and extended |
| `codex/luna-additive-20260915` | [#10](https://github.com/thefairyassociation/kissingnumbers/pull/10) | batched PyTorch 841 search and exact decimal certificates | merged in full under `lower_bound/experimental/` |
| `claude/kissing-number-dim-13-97lh52` | [#2](https://github.com/thefairyassociation/kissingnumbers/pull/2) (merged early) | 12 commits pushed *after* the PR merged: both dimension-14 leads closed, optimiser-branch review | merged; see "Partly taken" below |
| `codex/d13-packingstar-exact` | [#5](https://github.com/thefairyassociation/kissingnumbers/pull/5) | exact reconstruction of the PackingStar 1146-point configurations; ZE99 auxiliary-layer searches | merged in full |
| `codex/d13-optimal-code-scan` | – | same commit as `codex/d13-packingstar-exact` | nothing separate to merge |
| `codex/d13-steiner-switch-record` | [#9](https://github.com/thefairyassociation/kissingnumbers/pull/9) | audit of the claimed Steiner-switch family; 25-point auxiliary search | merged in full |
| `codex/d13-steiner-switch-run` | [#6](https://github.com/thefairyassociation/kissingnumbers/pull/6) | first (pre-audit) run of the Steiner-switch search | merged; the audited result files were kept |
| `codex/d13-joint-layer-169`, `-run` | [#7](https://github.com/thefairyassociation/kissingnumbers/pull/7) | CP-SAT search for a 169-point half-height layer | merged; **question now settled**, see below |
| `codex/d13-drop-tetrad-run` | [#8](https://github.com/thefairyassociation/kissingnumbers/pull/8) | one-tetrad-deletion 177-layer searches | merged in full |
| `codex/d13-drop-tetrad-177` | – | a second CI run of the same seven searches | recorded as merged; duplicate, nothing taken |
| `codex/n841-faithful-cpu` | [#3](https://github.com/thefairyassociation/kissingnumbers/pull/3) (merged) | source-faithful 841 calibration tooling | was already on `main` |
| `cursor/kissing-dim12-de82`, `-dim13-`, `-dim14-` | – | the original per-dimension baselines | recorded as merged; superseded, nothing taken |
| `cursor/optimizer-blas-lbfgs-e452` | – | BLAS / L-BFGS rewrite of the optimiser | recorded as merged; reviewed and reworked into `main` by PR #3 |

"Recorded as merged" means a merge with git's `ours` strategy: the branch's
commits become ancestors of the consolidated history, so git reports the
branch as fully merged and deleting it loses nothing, but no file content was
taken from it.

## Partly taken

**`claude/kissing-number-dim-13-97lh52`.** PR #2 merged an early state of this
branch; twelve later commits never reached `main`. PR #3 then re-did the
optimiser part of that work on `main` after a review. So:

* taken: the closure of both dimension-14 leads (`dim14/dimv6.py`,
  `diag_dimv6.py`, `dimv6_verdict.py`, `why_global_v.py` and the report
  section "Both dimension-14 leads above are now ruled out"), the review of
  the optimiser branch (`docs/REVIEW_optimizer_branch.md`), and the
  progress-log entries;
* not taken: its versions of `riesz.c`, the Makefile, `selftest.sh`,
  `multistart_d12_841.py`, `run_d12_841.sh` and the calibration notes, all of
  which PR #3 superseded;
* dropped afterwards: its SessionStart hook (`.claude/`), `lib/briefing.sh`
  and `.reviewed`. The hook pointed at a directory name the repository no
  longer has, built one tool with `-ffast-math`, and restarted background
  searches at the start of every session.

**`codex/d13-steiner-switch-run`.** Its result files claimed 528
"one-conflict" replacement rows. PR #9 showed those are duplicate copies of
264 existing ZE99 diamonds, so the audited files from
`codex/d13-steiner-switch-record` were kept wherever the two disagreed.

## What the codex CI searches actually concluded

The codex branches ran their searches inside GitHub Actions and committed the
output back to the branch. Two of those runs never recorded a result, and
one family of results is weaker than its title suggests:

| search | PR | recorded outcome |
| --- | --- | --- |
| joint 169-point layer (`search_ze99_joint_layer.py`) | #7 | CI ran 25 minutes and failed in the commit step; nothing was recorded. **Now settled without a search:** `lower_bound/dim13/prove_joint_layer_168.py` proves the maximum is 168. |
| 25-point auxiliary layer (`search_ze99_aux_continuous.py`, matrix) | #9 | all four CI jobs failed within two minutes; nothing was recorded. Not re-run. |
| one-tetrad deletion, 177-point layer | #8 | all seven orbit representatives finished, but the main ("signed") case ended `UNKNOWN` at the 22-minute time limit in every one. Those were time-outs, not exclusions. **Now settled by a bound instead of a search:** `lower_bound/dim13/prove_drop_tetrad_176.py` shows a layer has at most 176 points after any deletion, so this family cannot pass 1154. |
| Steiner-switch family, integer shell | #9 | exhaustive over 8192 candidates: no 1155-point witness in that family. |
| PackingStar 1146 reconstruction | #5 | exact rational certificates for both published 1146-point configurations (they do not beat 1154). |

## Things removed in the cleanup

All of these remain in git history.

* `1.out` … `7.out`, `riesz_1.out`, `riesz_5.out` at the top level: optimiser
  dumps from the first sessions. Nothing read them.
* Seven compiled binaries in `lib/`. They are build products; `make -C
  lower_bound/lib all` rebuilds them.
* Thirteen single-use workflow files, one per codex search, each hard-wired to
  a `codex/*` branch name, and their three `*_TRIGGER.md` marker files. They
  are replaced by one reusable workflow, `.github/workflows/search.yml`, driven
  by `.github/search-campaign.json`.
* Per-round stream logs of the August search loops (removed on the
  `claude/kissing-number-dim-13-97lh52` branch itself; outcomes are in
  `lower_bound/logs/progress.log`).

## Suggested housekeeping on GitHub

Once the consolidated branch is merged into `main`:

1. close PRs #4, #5, #6, #7, #8, #9 and #10 (their content is in `main`);
2. delete every branch in the table above.

Neither step loses anything.

# Reproduction guide

Run commands from the repository root. All paths are repository-relative or derived from the script location; no original workstation path is required.

## 1. Verify the delivered files

```bash
python scripts/verify_package.py
```

This checks the SHA-256 inventory of the manuscript, scientific source, protocols, and evidence. Hashes detect byte changes; they are not an external digital signature.

## 2. Replay the manuscript's numerical summaries

```bash
python scripts/reproduce.py --output outputs/reproduction
```

This command needs only Python's standard library. It recomputes signed benefits, per-severity means and sample standard deviations, and failure-inclusive readiness counts from the bundled saved values. It verifies the saved comparison denominators and totals of 51 model requests and 132 tool calls. Incomplete attempts remain included.

| Manuscript item | Evidence |
|---|---|
| Table I | `evidence/datasets.json` |
| Table II and Figure 2 | `evidence/missingness.json` (27 draws) |
| Table III | `evidence/adult_examples.json` (eight declared illustrations) |
| Table IV and Figure 3 | `evidence/readiness.json`, native C01 rules-repair scores |
| Table V and Figure 4 | `evidence/readiness.json`, six cases and 12 attempts |

Positive benefit means repaired minus observed score for the higher-is-better metrics used in these tables. Percentage points equal 100 times the score difference. Recovery is undefined without positive degradation or valid utility; the exact target-proxy diagnostic is not a valid generalization result. Three-draw sample SD describes perturbation variation, not a confidence interval.

## 3. Redraw the evidence figures

Install the toolkit with the `figures` extra, then run:

```bash
python scripts/plot_results.py --output outputs/figures
```

The figures present the same saved measurements. They are portable redraws, not byte-identical reproductions of the manuscript's typesetting. The original manuscript remains unchanged.

## 4. Exercise the deterministic components

```bash
python examples/offline_readiness.py
python -m pytest
```

The demonstration constructs synthetic input and exercises the original six-case generator, rules decision, canonical repair, and protected-value checks. It does not train a model, make a provider request, or represent a paper observation. The tests include original unit tests and checks against manuscript values.

## Boundaries of this release

The package supports numerical replay from saved aggregates, inspection of prompts and policy, and reuse of deterministic components. It does not include live-provider execution, full experiment orchestration, original row-level datasets, model checkpoints, or all saved prediction arrays. It cannot reconstruct every historical model fit or bootstrap interval from these files alone. Exact historical replay of those experiments requires the fuller research archive.

The original dataset acquisition and transformation functions are included. They verify pinned input hashes and should be run only when deliberately acquiring public data; [DATA.md](DATA.md) explains their use. Upstream metadata serialization can change and cause a checksum rejection. Do not bypass that rejection or relabel new bytes as the original inputs.

Historical protocols are separate: task-specific median/mode repair, the earlier domain-only pilot, and the combined readiness policy are not interchangeable treatments. The saved figures and tables do not assert a controlled architectural comparison, external validation, or clinical applicability.

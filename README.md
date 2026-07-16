# Hebbian Interference Geometry in Hopfield Networks

Reproducibility materials for the manuscript:

**Hebbian Interference Geometry in Hopfield Networks: Finite-Size Writing
Costs, Retrieval Risk, and Conditional Non-Reversibility**

Author: **Chikoo Oosawa**  
Kyushu Institute of Technology  
ORCID: [0009-0007-8824-5465](https://orcid.org/0009-0007-8824-5465)

## Scope

This repository contains code, aggregated data, representative raw-data
samples, and figure files for a finite-size statistical-mechanical study of
Hebbian memory-writing interference in Hopfield networks.

The analyses include:

- exact conversion from pattern overlap to Hebbian-direction overlap;
- intensive and cumulative writing costs;
- robustness across correlated-memory ensembles;
- finite-size diagnostics;
- overlap-moment analytical prediction;
- retrieval-overlap and retrieval-failure-risk analyses;
- conditional Randers writing/erasing asymmetry.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash scripts/run_quick.sh
```

## Full retrieval experiment

```bash
bash scripts/run_retrieval_full.sh
```

The executed heavy design used:

- `N = 128, 256, 512`;
- `P/N = 0.05, 0.10, 0.15, 0.20, 0.30`;
- latent, block, and mixture correlated ensembles;
- correlation parameters `0.15, 0.25, 0.35, 0.45`;
- cue corruption probabilities `0.05, 0.10, 0.15, 0.20`;
- 50 realizations and 12 cues per noise level.

This produced 18,000 realization rows and 864,000 noisy-cue trial rows.

## Full raw data

The complete audit-ready archive is supplied as the GitHub Release asset:

`L4_Hebbian_Interference_Geometry_full_reproducibility_v1.0.zip`

SHA-256:

`e50600e3a4a5f1a3dd37de3ce2d531c5f3dd96cce1b58a6158e30a608e312f32`

Large trial-level tables are not stored in Git history. Aggregated results and
1,000-row schema samples are included under `data/`.

## Key numerical checks

The corrected overlap-moment prediction tracks the simulated excess intensive
cost across 180 conditions with approximately:

- Pearson correlation: `0.9992`;
- Spearman correlation: `0.9927`.

Across 720 correlated-minus-uncorrelated contrasts, the excess intensive cost
is a moderate diagnostic of increased retrieval-failure risk:

- pooled Spearman correlation: approximately `0.407`;
- for nominal load `P/N <= 0.20`: approximately `0.552`.

These are statistical associations, not deterministic recall predictions or a
new learning algorithm.

## Repository structure

```text
src/                 active Python simulations
scripts/             quick and full execution commands
docs/                manuals and data dictionary
data/aggregated/     publication-level summaries
data/samples/        representative raw-table samples
figures/main/        main manuscript figures
figures/supplementary/ supplementary figures
release_assets/      checksum and release-asset instructions
paper/               associated-manuscript metadata
```

## Reproducibility notes

- Python 3.10 or later is supported; Python 3.11 is recommended.
- Active scripts use NumPy and Matplotlib only.
- Figures are generated as PNG to avoid platform-dependent PDF-backend issues.
- Long retrieval runs support checkpointing and resume.
- Random seeds and executed parameter settings are recorded in the scripts and
  manuals.
  
## Citation

Please cite the fixed repository release using the metadata provided in
`CITATION.cff`.

## License

Source code is distributed under the MIT License.

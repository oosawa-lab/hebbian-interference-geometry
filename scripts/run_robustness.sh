#!/usr/bin/env bash
set -euo pipefail
python3 src/robustness_simulation.py \
  --outdir results/robustness \
  --Ns 64 128 256 512 \
  --loads 0.10 0.30 0.60 \
  --generators latent block mixture biased \
  --corrs 0.15 0.25 0.35 0.45 \
  --lambdas 0.5 1.0 2.0 4.0 \
  --randers-alpha 0.15 \
  --randers-modes mean pc1 random-fixed \
  --reps 80 \
  --bootstrap 500

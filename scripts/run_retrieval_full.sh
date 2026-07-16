#!/usr/bin/env bash
set -euo pipefail
python3 src/retrieval_geometry_simulation.py \
  --outdir results/retrieval_geometry \
  --Ns 128 256 512 \
  --loads 0.05 0.10 0.15 0.20 0.30 \
  --generators latent block mixture \
  --corrs 0.15 0.25 0.35 0.45 \
  --cue-noises 0.05 0.10 0.15 0.20 \
  --reps 50 \
  --cues-per-noise 12 \
  --max-steps 25 \
  --lambda-interference 2.0

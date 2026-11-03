#!/usr/bin/env bash
# Section 2.3, Figure 7: SC alone at staleness 63 with batch 32, 64 (two seeds) and 512.
# The on-policy baseline at batch 512 is in sec2_2_large_batch.sh.
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=st63-b32-sc-s1  N=64 B=32  CAP=0 SC=true                      bash scripts/run.sh
NAME=st63-b64-sc-s1  N=64 B=64  CAP=0 SC=true                      bash scripts/run.sh
NAME=st63-b64-sc-s2  N=64 B=64  CAP=0 SC=true  SEED=2              bash scripts/run.sh
NAME=st63-b512-sc-s1 N=64 B=512 CAP=0 SC=true            EPOCHS=12  bash scripts/run.sh

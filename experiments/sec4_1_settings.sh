#!/usr/bin/env bash
# Section 4.1, Figure 11: SC alone at staleness 63 and batch 32, rescued by a larger batch (512) or a lower learning rate
# (0.01), and at batch 64 by less staleness (63 to 15). Its other runs are st63-b32-sc-s1, st63-b512-sc-s1,
# st63-b64-sc-s1 and st63-b64-sc-s2 (sec2_3_small_batch.sh), and st15-b64-sc-s1 (sec2_1_sc_is_grid.sh).
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=st63-b32-lr0.01-sc-s1 N=64 B=32 LR=0.01 CAP=0 SC=true  bash scripts/run.sh

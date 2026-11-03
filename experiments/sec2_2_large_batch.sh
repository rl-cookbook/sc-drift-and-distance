#!/usr/bin/env bash
# Section 2.2, Figures 5 and 6: batch 512 against batch 64 at staleness 15, and the on-policy baselines.
# Figure 5: vanilla IS, SC alone and SC + TIS. Figure 6: the first rollout without correction and with SC alone.
# The batch-64 runs are in sec2_1_sc_is_grid.sh.
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=st15-b512-is-s1     N=16 B=512 CAP=1e9 SC=false  bash scripts/run.sh
NAME=st15-b512-sc-s1     N=16 B=512 CAP=0   SC=true   bash scripts/run.sh
NAME=st15-b512-sc-tis-s1 N=16 B=512 CAP=2   SC=true   bash scripts/run.sh
NAME=st15-b512-none-s1   N=16 B=512 CAP=0   SC=false  bash scripts/run.sh
NAME=onpolicy-b64-s1     N=1  B=64  CAP=0   SC=true   bash scripts/run.sh
NAME=onpolicy-b512-s1    N=1  B=512 CAP=0   SC=true   bash scripts/run.sh

#!/usr/bin/env bash
# Section 2.1, Figure 4: score centering (SC) and importance sampling, each on and off.
# 0.8B on GSM8K, staleness 15, batch 64. CAP=0 is no IS, CAP=2 TIS, CAP=1e9 vanilla IS.
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=st15-b64-none-s1   N=16 B=64 CAP=0   SC=false  bash scripts/run.sh
NAME=st15-b64-is-s1     N=16 B=64 CAP=1e9 SC=false  bash scripts/run.sh
NAME=st15-b64-tis-s1    N=16 B=64 CAP=2   SC=false  bash scripts/run.sh
NAME=st15-b64-sc-s1     N=16 B=64 CAP=0   SC=true   bash scripts/run.sh
NAME=st15-b64-sc-is-s1  N=16 B=64 CAP=1e9 SC=true   bash scripts/run.sh
NAME=st15-b64-sc-tis-s1 N=16 B=64 CAP=2   SC=true   bash scripts/run.sh

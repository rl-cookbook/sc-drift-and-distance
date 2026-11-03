#!/usr/bin/env bash
# Section 3, Figure 10: the KL at batch 128 over the KL at batch 1024, in one rollout at staleness 31 with SC alone
# (three seeds, each run twice). The last 18 runs are two rollouts of SC alone at staleness 3, 15 and 63, learning rate
# 0.015, 0.03 and 0.06, and batch 64 and 512. The two-term formula of Figure 10 and of the table in Section 4.1 is
# fitted on them.
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=st31-b128-probe-sc-s3         N=32 B=128           CAP=0 SC=true OPT_STEPS=32  SEED=3 EVAL=false  bash scripts/run.sh
NAME=st31-b128-probe-sc-s3-rerun   N=32 B=128           CAP=0 SC=true OPT_STEPS=32  SEED=3 EVAL=false  bash scripts/run.sh
NAME=st31-b128-probe-sc-s4         N=32 B=128           CAP=0 SC=true OPT_STEPS=32  SEED=4 EVAL=false  bash scripts/run.sh
NAME=st31-b128-probe-sc-s4-rerun   N=32 B=128           CAP=0 SC=true OPT_STEPS=32  SEED=4 EVAL=false  bash scripts/run.sh
NAME=st31-b128-probe-sc-s5         N=32 B=128           CAP=0 SC=true OPT_STEPS=32  SEED=5 EVAL=false  bash scripts/run.sh
NAME=st31-b128-probe-sc-s5-rerun   N=32 B=128           CAP=0 SC=true OPT_STEPS=32  SEED=5 EVAL=false  bash scripts/run.sh
NAME=st31-b1024-probe-sc-s3        N=32 B=1024          CAP=0 SC=true OPT_STEPS=32  SEED=3 EVAL=false  bash scripts/run.sh
NAME=st31-b1024-probe-sc-s3-rerun  N=32 B=1024          CAP=0 SC=true OPT_STEPS=32  SEED=3 EVAL=false  bash scripts/run.sh
NAME=st31-b1024-probe-sc-s4        N=32 B=1024          CAP=0 SC=true OPT_STEPS=32  SEED=4 EVAL=false  bash scripts/run.sh
NAME=st31-b1024-probe-sc-s4-rerun  N=32 B=1024          CAP=0 SC=true OPT_STEPS=32  SEED=4 EVAL=false  bash scripts/run.sh
NAME=st31-b1024-probe-sc-s5        N=32 B=1024          CAP=0 SC=true OPT_STEPS=32  SEED=5 EVAL=false  bash scripts/run.sh
NAME=st31-b1024-probe-sc-s5-rerun  N=32 B=1024          CAP=0 SC=true OPT_STEPS=32  SEED=5 EVAL=false  bash scripts/run.sh
NAME=st3-b64-lr0.015-probe-sc-s1   N=4  B=64   LR=0.015 CAP=0 SC=true OPT_STEPS=8          EVAL=false  bash scripts/run.sh
NAME=st3-b512-lr0.015-probe-sc-s1  N=4  B=512  LR=0.015 CAP=0 SC=true OPT_STEPS=8          EVAL=false  bash scripts/run.sh
NAME=st3-b64-probe-sc-s1           N=4  B=64            CAP=0 SC=true OPT_STEPS=8          EVAL=false  bash scripts/run.sh
NAME=st3-b512-probe-sc-s1          N=4  B=512           CAP=0 SC=true OPT_STEPS=8          EVAL=false  bash scripts/run.sh
NAME=st3-b64-lr0.06-probe-sc-s1    N=4  B=64   LR=0.06  CAP=0 SC=true OPT_STEPS=8          EVAL=false  bash scripts/run.sh
NAME=st3-b512-lr0.06-probe-sc-s1   N=4  B=512  LR=0.06  CAP=0 SC=true OPT_STEPS=8          EVAL=false  bash scripts/run.sh
NAME=st15-b64-lr0.015-probe-sc-s1  N=16 B=64   LR=0.015 CAP=0 SC=true OPT_STEPS=32         EVAL=false  bash scripts/run.sh
NAME=st15-b512-lr0.015-probe-sc-s1 N=16 B=512  LR=0.015 CAP=0 SC=true OPT_STEPS=32         EVAL=false  bash scripts/run.sh
NAME=st15-b64-probe-sc-s1          N=16 B=64            CAP=0 SC=true OPT_STEPS=32         EVAL=false  bash scripts/run.sh
NAME=st15-b512-probe-sc-s1         N=16 B=512           CAP=0 SC=true OPT_STEPS=32         EVAL=false  bash scripts/run.sh
NAME=st15-b64-lr0.06-probe-sc-s1   N=16 B=64   LR=0.06  CAP=0 SC=true OPT_STEPS=32         EVAL=false  bash scripts/run.sh
NAME=st15-b512-lr0.06-probe-sc-s1  N=16 B=512  LR=0.06  CAP=0 SC=true OPT_STEPS=32         EVAL=false  bash scripts/run.sh
NAME=st63-b64-lr0.015-probe-sc-s1  N=64 B=64   LR=0.015 CAP=0 SC=true OPT_STEPS=128        EVAL=false  bash scripts/run.sh
NAME=st63-b512-lr0.015-probe-sc-s1 N=64 B=512  LR=0.015 CAP=0 SC=true OPT_STEPS=128        EVAL=false  bash scripts/run.sh
NAME=st63-b64-probe-sc-s1          N=64 B=64            CAP=0 SC=true OPT_STEPS=128        EVAL=false  bash scripts/run.sh
NAME=st63-b512-probe-sc-s1         N=64 B=512           CAP=0 SC=true OPT_STEPS=128        EVAL=false  bash scripts/run.sh
NAME=st63-b64-lr0.06-probe-sc-s1   N=64 B=64   LR=0.06  CAP=0 SC=true OPT_STEPS=128        EVAL=false  bash scripts/run.sh
NAME=st63-b512-lr0.06-probe-sc-s1  N=64 B=512  LR=0.06  CAP=0 SC=true OPT_STEPS=128        EVAL=false  bash scripts/run.sh

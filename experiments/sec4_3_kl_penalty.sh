#!/usr/bin/env bash
# Section 4.3, Figure 16: KL penalties against a lower learning rate at staleness 63 and batch 32, with SC. A constant
# beta of 0.01 (the Tinker default, two seeds), 0.075 and 0.3, and the learning rate cut to a third (0.01) with no
# penalty (two seeds). The penalty that decays with token age (Zhang, 2026), beta0 / sqrt(k) on mini-batch k, with beta0
# 0.03, 0.1 and 0.3 (two seeds), and with beta0 0.3 and the exact KL over the sampler's top-32 tokens. SC alone is
# st63-b32-sc-s1 (sec2_3_small_batch.sh) and the first lr 0.01 seed is st63-b32-lr0.01-sc-s1 (sec4_1_settings.sh).
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=st63-b32-sc-kl0.01-s1           N=64 B=32         CAP=0 SC=true KL_BETA=0.01                                                 bash scripts/run.sh
NAME=st63-b32-sc-kl0.01-s2           N=64 B=32         CAP=0 SC=true KL_BETA=0.01                                         SEED=2  bash scripts/run.sh
NAME=st63-b32-sc-kl0.075-s1          N=64 B=32         CAP=0 SC=true KL_BETA=0.075                                                bash scripts/run.sh
NAME=st63-b32-sc-kl0.3-s1            N=64 B=32         CAP=0 SC=true KL_BETA=0.3                                                  bash scripts/run.sh
NAME=st63-b32-lr0.01-sc-s2           N=64 B=32 LR=0.01 CAP=0 SC=true                                                      SEED=2  bash scripts/run.sh
NAME=st63-b32-sc-kldecay0.03-s1      N=64 B=32         CAP=0 SC=true KL_BETA=0.03  KL_SCHEDULE=inv_sqrt_age                       bash scripts/run.sh
NAME=st63-b32-sc-kldecay0.1-s1       N=64 B=32         CAP=0 SC=true KL_BETA=0.1   KL_SCHEDULE=inv_sqrt_age                       bash scripts/run.sh
NAME=st63-b32-sc-kldecay0.3-s1       N=64 B=32         CAP=0 SC=true KL_BETA=0.3   KL_SCHEDULE=inv_sqrt_age                       bash scripts/run.sh
NAME=st63-b32-sc-kldecay0.3-s2       N=64 B=32         CAP=0 SC=true KL_BETA=0.3   KL_SCHEDULE=inv_sqrt_age               SEED=2  bash scripts/run.sh
NAME=st63-b32-sc-kldecay0.3-exact-s1 N=64 B=32         CAP=0 SC=true KL_BETA=0.3   KL_SCHEDULE=inv_sqrt_age KL_EXACT=true         bash scripts/run.sh

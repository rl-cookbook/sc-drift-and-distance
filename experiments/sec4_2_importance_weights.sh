#!/usr/bin/env bash
# Section 4.2, Figures 12 to 15: importance weights on top of SC.
# Figure 12: an fp8 sampler (weights and activations) at batch 512 and staleness 15, and vanilla IS at staleness 0. The
#   fp8kv runs also store the KV cache in fp8. The bf16 on-policy baseline is onpolicy-b512-s1 (sec2_2_large_batch.sh).
# Figure 13: SC alone against vanilla IS at staleness 63 and batch 512, two seeds each. SC seed 1 is st63-b512-sc-s1
#   (sec2_3_small_batch.sh).
# Figure 14: the first rollout (64 steps) of SC alone, SC + TIS, SC + min(1, w) (BRAKE=true) and vanilla IS at
#   staleness 63 and batch 512, two seeds each.
# Figure 15: SC + min(1, w) over whole runs at staleness 63, against st63-b64-sc-s1 and st63-b32-sc-s1
#   (sec2_3_small_batch.sh).
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=onpolicy-b512-fp8-is-s1     N=1  B=512 CAP=1e9 SC=false            FP8=true                                                        bash scripts/run.sh
NAME=st15-b512-fp8-is-s1         N=16 B=512 CAP=1e9 SC=false            FP8=true                                                        bash scripts/run.sh
NAME=st15-b512-fp8-sc-tis-s1     N=16 B=512 CAP=2   SC=true             FP8=true                                                        bash scripts/run.sh
NAME=st15-b512-fp8-sc-s1         N=16 B=512 CAP=0   SC=true             FP8=true                                                        bash scripts/run.sh
NAME=onpolicy-b512-fp8kv-is-s1   N=1  B=512 CAP=1e9 SC=false            FP8=true FP8_KV=true                                            bash scripts/run.sh
NAME=st15-b512-fp8kv-is-s1       N=16 B=512 CAP=1e9 SC=false            FP8=true FP8_KV=true                                            bash scripts/run.sh
NAME=st15-b512-fp8kv-sc-tis-s1   N=16 B=512 CAP=2   SC=true             FP8=true FP8_KV=true                                            bash scripts/run.sh
NAME=st15-b512-fp8kv-sc-s1       N=16 B=512 CAP=0   SC=true             FP8=true FP8_KV=true                                            bash scripts/run.sh
NAME=st63-b512-sc-s2             N=64 B=512 CAP=0   SC=true                                                SEED=2 EPOCHS=12             bash scripts/run.sh
NAME=st63-b512-is-s1             N=64 B=512 CAP=1e9 SC=false                                 OPT_STEPS=320                              bash scripts/run.sh
NAME=st63-b512-is-s3             N=64 B=512 CAP=1e9 SC=false                                 OPT_STEPS=320 SEED=3                       bash scripts/run.sh
NAME=st63-b512-first-sc-s1       N=64 B=512 CAP=0   SC=true                                  OPT_STEPS=64                   EVAL=false  bash scripts/run.sh
NAME=st63-b512-first-sc-s2       N=64 B=512 CAP=0   SC=true                                  OPT_STEPS=64  SEED=2           EVAL=false  bash scripts/run.sh
NAME=st63-b512-first-sc-tis-s1   N=64 B=512 CAP=2   SC=true                                  OPT_STEPS=64                   EVAL=false  bash scripts/run.sh
NAME=st63-b512-first-sc-tis-s2   N=64 B=512 CAP=2   SC=true                                  OPT_STEPS=64  SEED=2           EVAL=false  bash scripts/run.sh
NAME=st63-b512-first-sc-brake-s1 N=64 B=512 CAP=0   SC=true  BRAKE=true                      OPT_STEPS=64                   EVAL=false  bash scripts/run.sh
NAME=st63-b512-first-sc-brake-s2 N=64 B=512 CAP=0   SC=true  BRAKE=true                      OPT_STEPS=64  SEED=2           EVAL=false  bash scripts/run.sh
NAME=st63-b512-first-is-s1       N=64 B=512 CAP=1e9 SC=false                                 OPT_STEPS=64                   EVAL=false  bash scripts/run.sh
NAME=st63-b512-first-is-s2       N=64 B=512 CAP=1e9 SC=false                                 OPT_STEPS=64  SEED=2           EVAL=false  bash scripts/run.sh
NAME=st63-b64-sc-brake-s1        N=64 B=64  CAP=0   SC=true  BRAKE=true                                                                 bash scripts/run.sh
NAME=st63-b32-sc-brake-s1        N=64 B=32  CAP=0   SC=true  BRAKE=true                                                                 bash scripts/run.sh

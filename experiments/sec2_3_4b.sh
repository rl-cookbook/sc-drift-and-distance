#!/usr/bin/env bash
# Section 2.3, Figures 8 and 9: Qwen3.5-4B-Base on DAPO-Math-17k at staleness 15, learning rate 0.033, batch 16 and 128,
# and SC + KL penalty (beta 0.01) at batch 16. Figure 9 scores the final checkpoints and the base model with
# scripts/eval_4b.py on AIME 2024, AIME 2025 and MATH-500.
# One run per line. Run the file to go through all of them, or copy a line.
set -e
cd "$(dirname "$0")/.."

NAME=4b-st15-b16-is-s1        MODEL=4B N=16 B=16  CAP=1e9 SC=false                             bash scripts/run.sh
NAME=4b-st15-b16-sc-tis-s1    MODEL=4B N=16 B=16  CAP=2   SC=true                              bash scripts/run.sh
NAME=4b-st15-b16-sc-s1        MODEL=4B N=16 B=16  CAP=0   SC=true                              bash scripts/run.sh
NAME=4b-st15-b16-sc-kl0.01-s1 MODEL=4B N=16 B=16  CAP=0   SC=true  KL_BETA=0.01                bash scripts/run.sh
NAME=4b-onpolicy-b128-s1      MODEL=4B N=1  B=128 CAP=0   SC=true               OPT_STEPS=160  bash scripts/run.sh
NAME=4b-st15-b128-is-s1       MODEL=4B N=16 B=128 CAP=1e9 SC=false              OPT_STEPS=320  bash scripts/run.sh
NAME=4b-st15-b128-sc-s1       MODEL=4B N=16 B=128 CAP=0   SC=true               OPT_STEPS=320  bash scripts/run.sh
NAME=4b-st15-b128-sc-tis-s1   MODEL=4B N=16 B=128 CAP=2   SC=true               OPT_STEPS=320  bash scripts/run.sh

CKPT=${CKPT_DIR:-$HOME/ckpts/drift}
cd SkyRL
uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt base                                          --out ../results/4b-base
uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt "$CKPT/4b-onpolicy-b128-s1/global_step_160"     --out ../results/4b-onpolicy-b128-s1
uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt "$CKPT/4b-st15-b128-is-s1/global_step_20"       --out ../results/4b-st15-b128-is-s1
uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt "$CKPT/4b-st15-b128-sc-s1/global_step_20"       --out ../results/4b-st15-b128-sc-s1
uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt "$CKPT/4b-st15-b128-sc-tis-s1/global_step_20"   --out ../results/4b-st15-b128-sc-tis-s1
uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt "$CKPT/4b-st15-b16-sc-kl0.01-s1/global_step_40" --out ../results/4b-st15-b16-sc-kl0.01-s1

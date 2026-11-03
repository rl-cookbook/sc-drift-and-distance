#!/usr/bin/env bash
# One training run of the blog on one node with 8 GPUs, from the SkyRL copy in ./SkyRL (see SkyRL/MODIFICATIONS.md).
#
# Staleness: each rollout is split into N mini-batches of B responses, with one SGD step on each before the sampler
# syncs, so mini-batch k is k steps stale (maximum staleness N - 1, and N = 1 is on-policy).
#
#   NAME       run name (W&B run and checkpoint directory)                     required
#   MODEL      0.8B (Qwen3.5-0.8B-Base on GSM8K) or 4B (Qwen3.5-4B-Base on DAPO-Math-17k)   0.8B
#   N          mini-batches per rollout                                         16
#   B          responses per optimizer step (8 per prompt)                      64
#   LR         SGD learning rate                                                0.03 (0.8B), 0.033 (4B)
#   CAP        importance weight: 0 none, 2 TIS, 1e9 vanilla IS                 0
#   SC         score centering                                                  true
#   BRAKE      SC with the low-end weight min(1, p/q) on negative advantages    false
#   KL_BETA    KL penalty toward the sampler                                    0
#   KL_SCHEDULE  constant, or inv_sqrt_age for KL_BETA / sqrt(k) on mini-batch k (none on k = 0)   constant
#   KL_EXACT   exact KL over the sampler's top-32 head instead of its k1 estimate   false
#   FP8        fp8 sampler (weights and activations), FP8_KV adds an fp8 KV cache   false
#   OPT_STEPS  optimizer steps                                                  640
#   SEED                                                                        1
#   EVAL       evaluate during training (0.8B: 500 GSM8K problems every 64 steps)   true (0.8B), false (4B)
#   EPOCHS     dataloader epochs, enough to cover OPT_STEPS                     10
#   DATA_DIR, CKPT_DIR, SKYRL_DIR, WANDB_PROJECT                                ~/data/drift, ~/ckpts/drift, ./SkyRL, drift-distance
set -euo pipefail
: "${NAME:?set NAME}"
MODEL=${MODEL:-0.8B}
DATA_DIR=${DATA_DIR:-$HOME/data/drift}
CKPT_DIR=${CKPT_DIR:-$HOME/ckpts/drift}
SKYRL_DIR=${SKYRL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/SkyRL}
case $MODEL in
  0.8B) POLICY=Qwen/Qwen3.5-0.8B-Base TRAIN=$DATA_DIR/train_gsm8k.parquet VAL="['$DATA_DIR/val_gsm8k500.parquet']"
        LR=${LR:-0.03} EVAL=${EVAL:-true} MICRO_FWD=4 ;;
  4B)   POLICY=Qwen/Qwen3.5-4B-Base TRAIN=$DATA_DIR/train.parquet VAL="['$DATA_DIR/val_aime25.parquet']"
        LR=${LR:-0.033} EVAL=${EVAL:-false} MICRO_FWD=2 ;;
  *)    echo "MODEL must be 0.8B or 4B" >&2; exit 1 ;;
esac
N=${N:-16}
B=${B:-64}
CAP=${CAP:-0}
SC=${SC:-true}
BRAKE=${BRAKE:-false}
KL_BETA=${KL_BETA:-0}
KL_SCHEDULE=${KL_SCHEDULE:-constant}
KL_EXACT=${KL_EXACT:-false}
FP8=${FP8:-false}
FP8_KV=${FP8_KV:-false}
OPT_STEPS=${OPT_STEPS:-640}
SEED=${SEED:-1}
EPOCHS=${EPOCHS:-10}
MAX_RESPONSE=4096

MB=$((B / 8))
MAX_STEPS=$(( (OPT_STEPS + N - 1) / N ))
EVAL_INTERVAL=$(( 64 / N )); [ "$EVAL_INTERVAL" -lt 1 ] && EVAL_INTERVAL=1
[ "$EVAL" = false ] && EVAL_INTERVAL=100000

KW='"gdn_prefill_backend": "triton"'
if [ "$FP8" = true ]; then KW="$KW, \"quantization\": \"fp8\""; fi
if [ "$FP8_KV" = true ]; then KW="$KW, \"kv_cache_dtype\": \"fp8\""; fi
EXTRA=()
if [ "$BRAKE" = true ]; then EXTRA+=(trainer.algorithm.score_centering.neg_brake=true); fi
if [ "$KL_BETA" != 0 ]; then
  EXTRA+=(trainer.algorithm.score_centering.kl_anchor_beta=$KL_BETA trainer.algorithm.score_centering.kl_anchor_batch=$B)
  if [ "$KL_SCHEDULE" != constant ]; then
    EXTRA+=(trainer.algorithm.score_centering.kl_anchor_schedule=$KL_SCHEDULE trainer.algorithm.score_centering.kl_anchor_minibatches=$N)
  fi
  if [ "$KL_EXACT" = true ]; then EXTRA+=(trainer.algorithm.score_centering.kl_anchor_estimator=exact_head); fi
fi

# Qwen3.5's gated delta-net layers run on fla's Triton kernels in all our runs (its TileLang kernels failed to compile).
export FLA_TILELANG=${FLA_TILELANG:-0}
cd "$SKYRL_DIR"
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$TRAIN']" \
  data.val_data="$VAL" \
  environment.env_class=aime \
  trainer.policy.model.path=$POLICY \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.inference_engine.num_engines=8 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs="{$KW}" \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.policy_loss_type=token_tis \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$CAP \
  trainer.algorithm.score_centering.enabled=$SC \
  trainer.algorithm.score_centering.top_k=32 \
  trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
  trainer.algorithm.max_seq_len=$MAX_RESPONSE \
  trainer.algorithm.use_kl_loss=false \
  trainer.train_batch_size=$((N * MB)) \
  trainer.policy_mini_batch_size=$MB \
  trainer.update_epochs_per_batch=1 \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FWD \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=2 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=0.7 \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE \
  trainer.policy.optimizer_config.optimizer=sgd \
  trainer.policy.optimizer_config.sgd_momentum=0.0 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.policy.optimizer_config.max_grad_norm=1.0e9 \
  trainer.remove_microbatch_padding=false \
  trainer.epochs=$EPOCHS \
  trainer.max_training_steps=$MAX_STEPS \
  trainer.eval_before_train=$EVAL \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.eval_batch_size=1024 \
  trainer.ckpt_interval=$(( (MAX_STEPS + 3) / 4 )) \
  trainer.ckpt_path=$CKPT_DIR/$NAME \
  trainer.max_ckpts_to_keep=1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=latest \
  trainer.seed=$SEED \
  trainer.logger=wandb \
  trainer.project_name=${WANDB_PROJECT:-drift-distance} \
  trainer.run_name=$NAME \
  "${EXTRA[@]}" "$@"

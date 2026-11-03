# Modifications

This directory is a copy of [SkyRL](https://github.com/NovaSky-AI/SkyRL) at upstream commit
`3d0ae22c1ba316f7211c973593de3f42ba040c6b`, licensed under Apache-2.0 (see `LICENSE`). We changed it in three ways.

## Score centering

Score centering comes from SkyRL [PR #2244](https://github.com/NovaSky-AI/SkyRL/pull/2244) by Eric Tang, merged onto
`3d0ae22`. Only the code is included, not the PR's tests and docs. vLLM returns the sampler's top-k log-probabilities
for every generated token, and `trainer.algorithm.score_centering.*` adds the centering term. It adds
`skyrl/backends/skyrl_train/utils/score_centering_support.py` and changes 25 other files under `skyrl/`.

## Our additions

These are in `skyrl/backends/skyrl_train/utils/ppo_utils.py`, `skyrl/backends/skyrl_train/distributed/fsdp_strategy.py`,
`skyrl/backends/skyrl_train/workers/worker.py`, `skyrl/train/config/config.py`, `skyrl/train/trainer.py` and
`skyrl/train/utils/utils.py`.

| Option | Default | What it does |
|---|---|---|
| `trainer.algorithm.policy_loss_type=token_tis` | | Token-level policy gradient against the sampler's log-probabilities, weighted by `f(p/q)`. |
| `trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high` | 2.0 | An upstream option that `token_tis` reads as its cap `C`. `C <= 0` gives no IS weight (`f = 1`), `C = 2` is TIS, and a very large `C` (`1e9`) is vanilla IS. Score centering uses the same `f`. |
| `trainer.algorithm.score_centering.neg_brake` | false | SC + min(1, w). On tokens with a negative advantage, the sampled and centering terms of SC are multiplied by `min(1, p/q)`. Only with `C <= 0`. |
| `trainer.algorithm.score_centering.kl_anchor_beta` | 0 | KL penalty toward the sampler. By default it is `beta * (log q - log p)` on every sampled token, as in Tinker's StableReinforce. |
| `trainer.algorithm.score_centering.kl_anchor_batch` | 0 | Responses per optimizer step, so the penalty is normalized like the loss (`seq_mean_token_sum_norm`). `scripts/run.sh` sets it. |
| `trainer.algorithm.score_centering.kl_anchor_schedule` | constant | `constant` is beta on every sampled token. `inv_sqrt_age` is `beta / sqrt(k)` on mini-batch `k` of a rollout and nothing on `k = 0`, the SFC+KL anchor of Zhang (2026). FSDP only. |
| `trainer.algorithm.score_centering.kl_anchor_minibatches` | 1 | Mini-batches per rollout, for the age `k` of `inv_sqrt_age`. `scripts/run.sh` sets it to `N`. |
| `trainer.algorithm.score_centering.kl_anchor_estimator` | k1 | `k1` is `log q - log p` on the sampled token. `exact_head` is the exact gradient of KL(q \|\| p) over the sampler's top-k head, with the tail modeled as in score centering. Needs score centering. |
| `trainer.policy.optimizer_config.optimizer` | adam | `sgd` for plain SGD (FSDP only). `sgd_momentum` adds momentum (0 in all runs). |

`token_tis` also logs, per micro-batch, `is_kl_mu_pi` (the mean of `w - 1 - log w` with `w = p/q`, the KL of the blog),
`is_w_max` (the largest ratio), `is_frac_lt0p5` and `is_frac_gt2` (the shares of ratios below 0.5 and above 2), and
`is_capped_frac` (the share of ratios above `C`, when `C > 0`). The trainer logs each of them again per mini-batch index as
`is_kNN_*`, where mini-batch `k` is `k` optimizer steps stale, together with `is_kNN_grad_norm`.

## Removed

`docs/public/images/`, the images of SkyRL's documentation website, is left out to keep this copy small.

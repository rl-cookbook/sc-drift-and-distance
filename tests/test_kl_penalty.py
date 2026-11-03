"""CPU checks of the KL penalty options of token_tis, score_centering.kl_anchor_schedule and kl_anchor_estimator (see
SkyRL/MODIFICATIONS.md), on small random inputs.

usage: cd SkyRL && uv run --isolated --extra fsdp --with pytest python -m pytest ../tests -q
"""
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import compute_score_centering_loss
from skyrl.backends.skyrl_train.utils.ppo_utils import token_tis_policy_loss
from skyrl.train.config.config import AlgorithmConfig, ScoreCenteringConfig
from skyrl.train.utils.utils import validate_score_centering_cfg

B, T, V, L, BATCH, BETA = 4, 10, 12, 4096, 32, 0.3


def inputs(seed=0):
    """Sampler logits q, trainer logits p = q + noise, tokens drawn from q, a loss mask."""
    g = torch.Generator().manual_seed(seed)
    logits_q = torch.randn(B, T, V, generator=g, dtype=torch.float64) * 2
    logits_p = (logits_q + 0.7 * torch.randn(B, T, V, generator=g, dtype=torch.float64)).requires_grad_(True)
    tok = torch.multinomial(logits_q.softmax(-1).view(-1, V), 1, generator=g).view(B, T, 1)
    adv = torch.randn(B, 1, generator=g, dtype=torch.float64).expand(B, T).contiguous()
    mask = (torch.rand(B, T, generator=g) > 0.15).double()
    return logits_q, logits_p, tok, adv, mask


def grad(x, beta=0.0, schedule="constant", estimator="k1", age=None):
    """Gradient of the SC token_tis loss (no IS weight) w.r.t. the trainer logits. The sampler head is the whole
    vocabulary, so the exact KL has no modeled tail."""
    logits_q, logits_p, tok, adv, mask = x
    logits_p.grad = None
    lq, lp = logits_q.log_softmax(-1), logits_p.log_softmax(-1)
    cfg = AlgorithmConfig()
    cfg.loss_reduction, cfg.max_seq_len = "seq_mean_token_sum_norm", L
    cfg.off_policy_correction.token_tis_ratio_clip_high = 0.0
    sc = cfg.score_centering
    sc.enabled, sc.kl_anchor_beta, sc.kl_anchor_batch = True, beta, BATCH
    sc.kl_anchor_schedule, sc.kl_anchor_estimator = schedule, estimator
    sampled = lq.gather(-1, tok).squeeze(-1).detach()
    kw = dict(log_probs=lp.gather(-1, tok).squeeze(-1), old_log_probs=sampled, advantages=adv, config=cfg,
              loss_mask=mask, rollout_logprobs=sampled, rollout_topk_logprobs=lq.detach(), topk_log_probs=lp,
              kl_anchor_age=age)
    loss, _ = token_tis_policy_loss(**kw)
    loss.backward()
    return logits_p.grad.clone()


def test_defaults_keep_the_constant_k1_penalty():
    sc = AlgorithmConfig().score_centering
    assert (sc.kl_anchor_schedule, sc.kl_anchor_estimator, sc.kl_anchor_minibatches) == ("constant", "k1", 1)
    x = inputs()
    # The constant schedule ignores the age.
    assert torch.equal(grad(x, BETA, age=5), grad(x, BETA))


def test_inv_sqrt_age_skips_fresh_tokens_and_divides_by_sqrt_age():
    x = inputs()
    base = grad(x)
    constant = grad(x, BETA) - base
    assert torch.equal(grad(x, BETA, "inv_sqrt_age", age=0), base)
    for age in (1, 4, 63):
        d = grad(x, BETA, "inv_sqrt_age", age=age) - base
        assert torch.allclose(d, constant / age**0.5, rtol=1e-6, atol=1e-14)
    with pytest.raises(AssertionError):
        grad(x, BETA, "inv_sqrt_age", age=None)


def test_exact_head_has_the_exact_kl_gradient():
    """From the cluster's kl_exact_test.py: with the whole vocabulary as the head, d/dz of
    -c * sum_v sg[q_v - rho p_v] log p_v must equal c * (p - q)."""
    torch.manual_seed(0)
    b, a, v, c = 2, 3, 6, 0.37
    z = torch.randn(b, a, v, requires_grad=True)
    q = torch.softmax(torch.randn(b, a, v), -1)
    mask = torch.ones(b, a)
    loss, _ = compute_score_centering_loss(
        -c * mask, torch.log_softmax(z, -1), q.log(), mask, weight_fn=lambda r: torch.ones_like(r)
    )
    loss.sum().backward()
    assert (z.grad - c * (torch.softmax(z.detach(), -1) - q)).abs().max().item() < 1e-5


def test_exact_head_in_token_tis():
    x = inputs()
    logits_q, logits_p, _, _, mask = x
    scale = BETA / (BATCH * L)
    expected = scale * mask.unsqueeze(-1) * (logits_p.detach().softmax(-1) - logits_q.softmax(-1))
    d = grad(x, BETA, estimator="exact_head") - grad(x)
    assert torch.allclose(d, expected, rtol=0, atol=1e-5 * scale)
    d = grad(x, BETA, "inv_sqrt_age", "exact_head", age=4) - grad(x)
    assert torch.allclose(d, expected / 2, rtol=0, atol=1e-5 * scale)


def test_option_checks():
    with pytest.raises(ValueError):
        ScoreCenteringConfig(kl_anchor_schedule="linear")
    with pytest.raises(ValueError):
        ScoreCenteringConfig(kl_anchor_estimator="k3")
    sc = ScoreCenteringConfig(kl_anchor_beta=BETA, kl_anchor_schedule="inv_sqrt_age", kl_anchor_minibatches=64)

    def cfg(mini_batch):
        algorithm = SimpleNamespace(score_centering=sc)
        trainer = SimpleNamespace(algorithm=algorithm, train_batch_size=256, policy_mini_batch_size=mini_batch)
        return SimpleNamespace(trainer=trainer)

    # 256 prompts in mini-batches of 4 is N = 64, as in scripts/run.sh with N=64 B=32.
    validate_score_centering_cfg(cfg(4))
    with pytest.raises(ValueError):
        validate_score_centering_cfg(cfg(16))
    sc.kl_anchor_estimator = "exact_head"
    with pytest.raises(ValueError):
        validate_score_centering_cfg(cfg(4))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

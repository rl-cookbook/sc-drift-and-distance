"""CPU checks of the token_tis loss (see SkyRL/MODIFICATIONS.md) on small random inputs.

usage: cd SkyRL && uv run --isolated --extra fsdp --with pytest python -m pytest ../tests -q
"""
import torch

from skyrl.backends.skyrl_train.utils.ppo_utils import token_tis_policy_loss
from skyrl.train.config.config import AlgorithmConfig

B, T, V, K, L = 6, 24, 80, 32, 4096


def inputs(spread=0.7, adv_sign=None, seed=0):
    """Sampler logits q, trainer logits p = q + spread * noise, tokens drawn from q, the sampler's top-K head."""
    g = torch.Generator().manual_seed(seed)
    logits_q = torch.randn(B, T, V, generator=g, dtype=torch.float64) * 2
    logits_p = (logits_q + spread * torch.randn(B, T, V, generator=g, dtype=torch.float64)).requires_grad_(True)
    tok = torch.multinomial(logits_q.softmax(-1).view(-1, V), 1, generator=g).view(B, T, 1)
    adv = torch.randn(B, 1, generator=g, dtype=torch.float64).expand(B, T).contiguous()
    if adv_sign is not None:
        adv = adv.abs() * adv_sign
    mask = (torch.rand(B, T, generator=g) > 0.15).double()
    return logits_q, logits_p, tok, adv, mask


def run(x, cap, sc, brake=False, beta=0.0):
    logits_q, logits_p, tok, adv, mask = x
    logits_p.grad = None
    lq, lp = logits_q.log_softmax(-1), logits_p.log_softmax(-1)
    ids = lq.topk(K, dim=-1).indices
    cfg = AlgorithmConfig()
    cfg.loss_reduction, cfg.max_seq_len = "seq_mean_token_sum_norm", L
    cfg.off_policy_correction.token_tis_ratio_clip_high = cap
    cfg.score_centering.enabled, cfg.score_centering.neg_brake = sc, brake
    cfg.score_centering.kl_anchor_beta, cfg.score_centering.kl_anchor_batch = beta, 64
    kw = dict(log_probs=lp.gather(-1, tok).squeeze(-1), old_log_probs=lq.gather(-1, tok).squeeze(-1).detach(), advantages=adv,
              config=cfg, loss_mask=mask, rollout_logprobs=lq.gather(-1, tok).squeeze(-1).detach())
    if sc:
        kw.update(rollout_topk_logprobs=lq.gather(-1, ids).detach(), topk_log_probs=lp.gather(-1, ids))
    loss, metrics = token_tis_policy_loss(**kw)
    loss.backward()
    return loss.detach(), logits_p.grad.clone(), metrics


def test_vanilla_is_with_sc_is_vanilla_is():
    x = inputs()
    a, b = run(x, 1e9, False), run(x, 1e9, True)
    assert torch.allclose(a[0], b[0], atol=1e-10) and torch.allclose(a[1], b[1], atol=1e-10)


def test_tis_is_vanilla_is_below_the_cap():
    x = inputs(spread=0.05)
    assert run(x, 1e9, False)[2]["is_w_max"] < 2.0
    a, b = run(x, 2.0, True), run(x, 1e9, True)
    assert torch.allclose(a[0], b[0], atol=1e-10) and torch.allclose(a[1], b[1], atol=1e-10)


def test_no_is_weight_differs_from_vanilla_is_off_policy():
    x = inputs()
    assert not torch.allclose(run(x, 0.0, False)[1], run(x, 1e9, False)[1])


def test_low_end_weight_only_acts_on_negative_advantages():
    x = inputs(adv_sign=1.0)
    a, b = run(x, 0.0, True), run(x, 0.0, True, brake=True)
    assert torch.allclose(a[0], b[0], atol=1e-10) and torch.allclose(a[1], b[1], atol=1e-10)
    x = inputs(adv_sign=-1.0)
    assert not torch.allclose(run(x, 0.0, True)[1], run(x, 0.0, True, brake=True)[1])


def test_kl_penalty_pulls_the_sampled_tokens_up():
    x = inputs()
    _, _, tok, _, mask = x
    d = run(x, 0.0, True, beta=0.3)[1] - run(x, 0.0, True)[1]
    # The k1 penalty beta * (log q - log p) on a sampled token lowers the loss as its log-probability rises.
    on_token = d.gather(-1, tok).squeeze(-1)
    assert (on_token[mask > 0] < 0).all() and torch.allclose(d.sum(-1), torch.zeros_like(d.sum(-1)), atol=1e-12)
    assert torch.allclose(d[mask == 0], torch.zeros_like(d[mask == 0]))


def test_ratio_statistics():
    x = inputs()
    logits_q, logits_p, tok, adv, mask = x
    _, _, m = run(x, 0.0, False)
    lw = (logits_p.log_softmax(-1) - logits_q.log_softmax(-1)).gather(-1, tok).squeeze(-1).detach()[mask > 0]
    w = lw.exp()
    # The statistics are computed in float32.
    assert abs(m["is_kl_mu_pi"] / (w - 1 - lw).mean().item() - 1) < 1e-6
    assert abs(m["is_w_max"] / w.max().item() - 1) < 1e-6
    assert abs(m["is_frac_lt0p5"] - (w < 0.5).double().mean().item()) <= 1.0 / w.numel()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

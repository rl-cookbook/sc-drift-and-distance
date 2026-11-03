"""Evaluate a 4B checkpoint (or the base model) on AIME 2024, AIME 2025 (16 samples per problem) and MATH-500 (4 samples),
temperature 1.0, top-p 0.7, up to 4,096 response tokens, scored with SkyRL's aime `compute_score`. Uses all 8 GPUs.

usage (from ./SkyRL, so that it runs in SkyRL's environment):
  cd SkyRL
  uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt ~/ckpts/drift/<run>/global_step_<n> --out ../results/<run>
  uv run --isolated --extra fsdp python ../scripts/eval_4b.py --ckpt base --out ../results/4b-base
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

MAXTOK = 4096
TOP_P = 0.7
GPUS = 8


def convert(ckpt, out, model):
    """FSDP shards -> the original model directory (which vLLM loads natively) with its text weights replaced by ours."""
    import torch
    from safetensors.torch import load_file, save_file
    vdir = os.path.join(out, "hf_vllm")
    if os.path.exists(os.path.join(vdir, "done")):
        return vdir
    files = sorted(glob.glob(os.path.join(ckpt, "policy", "model_world_size_*_rank_*.pt")),
                   key=lambda f: int(re.search(r"rank_(\d+)", f).group(1)))
    shards = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    full = {}
    for k, v in shards[0].items():
        parts = [sh[k].to_local() if hasattr(sh[k], "to_local") else sh[k] for sh in shards]
        full[k] = parts[0].contiguous() if tuple(parts[0].shape) == tuple(v.shape) else torch.cat(parts, 0).contiguous()
    # The config ties the output layer to the embedding and vLLM honours that, so the trainer's lm_head copy is not used.
    full.pop("lm_head.weight", None)
    from huggingface_hub import snapshot_download
    orig = snapshot_download(model)
    shutil.copytree(orig, vdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.safetensors"), symlinks=False)
    replaced = 0
    for wf in sorted(glob.glob(os.path.join(orig, "*.safetensors"))):
        sd = load_file(wf)
        for k, v in full.items():
            key = "model.language_model." + k[len("model."):] if k.startswith("model.") else k
            if key in sd:
                sd[key] = v.to(sd[key].dtype)
                replaced += 1
        save_file(sd, os.path.join(vdir, os.path.basename(wf)))
    assert replaced == len(full), f"replaced {replaced} of {len(full)} text tensors"
    open(os.path.join(vdir, "done"), "w").write(f"replaced {replaced} of {len(full)}")
    return vdir


def tasks(data_dir):
    import pandas as pd
    from datasets import load_dataset
    gt = lambda r: str((r["reward_model"] if isinstance(r["reward_model"], dict) else eval(r["reward_model"]))["ground_truth"])
    prefix = pd.read_parquet(f"{data_dir}/train.parquet").iloc[0]["prompt"][0]["content"].split("\n\n")[0] + "\n\n"
    out = []
    for year in (2024, 2025):
        seen = set()
        for _, r in pd.read_parquet(f"{data_dir}/raw/dapo/aime-{year}.parquet").iterrows():
            p = str(r["prompt"][0]["content"])
            if p not in seen:
                seen.add(p)
                out.append({"ds": f"aime{year}", "id": len(seen) - 1, "prompt": p, "gt": gt(r), "n": 16})
        assert len(seen) == 30, (year, len(seen))
    for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test")):
        out.append({"ds": "math500", "id": i, "prompt": prefix + r["problem"], "gt": r["answer"], "n": 4})
    return out


def worker(args, shard):
    from skyrl_gym.envs.aime.utils import compute_score
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    items = [json.loads(line) for line in open(f"{args.out}/tasks.jsonl")][shard::GPUS]
    tok = AutoTokenizer.from_pretrained(args.model)
    path = args.model if args.ckpt == "base" else f"{args.out}/hf_vllm"
    llm = LLM(model=path, language_model_only=True, gdn_prefill_backend="triton", gpu_memory_utilization=0.85,
              max_model_len=MAXTOK + 2048, seed=shard)
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True, tokenize=False)
               for it in items]
    params = [SamplingParams(n=it["n"], temperature=1.0, top_p=TOP_P, max_tokens=MAXTOK, seed=shard * 100000 + k)
              for k, it in enumerate(items)]
    with open(f"{args.out}/res_{shard}.jsonl", "w") as f:
        for it, r in zip(items, llm.generate(prompts, params)):
            accs = [bool(compute_score(o.text, it["gt"])["acc"]) for o in r.outputs]
            f.write(json.dumps({"ds": it["ds"], "id": it["id"], "acc": accs}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="global_step_<n> directory of a run, or 'base'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/data/drift"))
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.shard is not None:
        return worker(args, args.shard)
    os.makedirs(args.out, exist_ok=True)
    if args.ckpt != "base":
        convert(args.ckpt, args.out, args.model)
    with open(f"{args.out}/tasks.jsonl", "w") as f:
        for t in tasks(args.data_dir):
            f.write(json.dumps(t) + "\n")
    procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--shard", str(i)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i in range(GPUS)]
    assert all(p.wait() == 0 for p in procs), "a worker failed"
    rows = [json.loads(line) for i in range(GPUS) for line in open(f"{args.out}/res_{i}.jsonl")]
    res = {}
    for ds in ("aime2024", "aime2025", "math500"):
        accs = [a for r in rows if r["ds"] == ds for a in r["acc"]]
        res[ds] = round(100 * sum(accs) / len(accs), 1)
    json.dump(res, open(f"{args.out}/scores.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()

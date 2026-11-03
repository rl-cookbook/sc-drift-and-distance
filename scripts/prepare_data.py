"""Build the training and evaluation parquets from SkyRL's GSM8K and DAPO-Math downloads (scripts/prepare_data.sh).

train_gsm8k.parquet   GSM8K train in the DAPO prompt format (0.8B runs)
val_gsm8k500.parquet  500 GSM8K test problems, same format (0.8B evaluation during training)
train.parquet         DAPO-Math-17k (cleaned) minus 500 held-out problems (4B runs)
val_dapo500.parquet   the 500 held-out DAPO problems
val_aime25.parquet    AIME 2025 (MathArena/aime_2025), each problem repeated 8 times
raw/dapo/aime-2025.parquet  AIME 2025 in the format of SkyRL's aime-2024.parquet (also read by eval_4b.py)

usage: python prepare_data.py <raw_dir> <out_dir>      (raw_dir holds gsm8k/ and dapo/ from prepare_data.sh)
"""
import ast
import os
import sys

import pandas as pd
from datasets import load_dataset

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
PROMPT = ("Solve the following math problem step by step. The last line of your response should be of the form "
          "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{problem}\n\n"
          "Remember to put your answer on its own line after \"Answer:\".")
pd.DataFrame([{"data_source": "aime2025", "prompt": [{"content": PROMPT.format(problem=ex["problem"]), "role": "user"}],
               "ability": "MATH", "reward_model": {"ground_truth": str(ex["answer"]), "style": "rule-lighteval/MATH_v2"},
               "extra_info": {"index": i, "raw_problem": ex["problem"], "split": "aime-2025"}}
              for i, ex in enumerate(load_dataset("MathArena/aime_2025", split="train"))]).to_parquet(f"{SRC}/dapo/aime-2025.parquet")


def as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def row(source, content, truth, index, split):
    return {"data_source": source, "prompt": [{"content": content, "role": "user"}], "ability": "MATH",
            "reward_model": {"ground_truth": str(truth), "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"index": str(index), "split": split}}


dapo = pd.read_parquet(f"{SRC}/dapo/dapo-math-17k-cleaned.parquet").sample(frac=1.0, random_state=0).reset_index(drop=True)
held, train = dapo.iloc[:500].copy(), dapo.iloc[500:].copy()
train.to_parquet(f"{OUT}/train.parquet")
prefix = dapo.iloc[0]["prompt"][0]["content"].split("\n\n")[0] + "\n\n"

# The eval sets are concatenated by SkyRL, so they share one schema exactly.
pd.DataFrame([row("dapo_heldout", str(r["prompt"][0]["content"]), as_dict(r["reward_model"])["ground_truth"], i, "dapo_heldout")
              for i, r in held.reset_index(drop=True).iterrows()]).to_parquet(f"{OUT}/val_dapo500.parquet")
aime = pd.concat([pd.read_parquet(f"{SRC}/dapo/aime-2025.parquet")] * 8, ignore_index=True)
pd.DataFrame([row("aime2025", str(r["prompt"][0]["content"]), as_dict(r["reward_model"])["ground_truth"], i, "aime2025")
              for i, r in aime.iterrows()]).to_parquet(f"{OUT}/val_aime25.parquet")

gsm = pd.read_parquet(f"{SRC}/gsm8k/validation.parquet").sample(n=500, random_state=0).reset_index(drop=True)
pd.DataFrame([row("gsm8k_test", prefix + as_dict(r["extra_info"])["question"], as_dict(r["reward_spec"])["ground_truth"], i,
                  "gsm8k_test") for i, r in gsm.iterrows()]).to_parquet(f"{OUT}/val_gsm8k500.parquet")
gtrain = pd.read_parquet(f"{SRC}/gsm8k/train.parquet").reset_index(drop=True)
pd.DataFrame([row("gsm8k_train", prefix + as_dict(r["extra_info"])["question"], as_dict(r["reward_spec"])["ground_truth"], i,
                  "gsm8k_train") for i, r in gtrain.iterrows()]).to_parquet(f"{OUT}/train_gsm8k.parquet")

for f in sorted(os.listdir(OUT)):
    print(f, len(pd.read_parquet(f"{OUT}/{f}")))

import pandas as pd
import os

RED_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

df_red   = pd.read_csv(RED_URL,   sep=";")
df_white = pd.read_csv(WHITE_URL, sep=";")

df_red["wine_type"]   = 0
df_white["wine_type"] = 1

df = pd.concat([df_red, df_white], ignore_index=True)

# Rename columns: replace spaces with underscores so all CSVs use the
# canonical 12-feature names (fixed_acidity, volatile_acidity, ...)
df.columns = [c.replace(" ", "_") for c in df.columns]

df = df.sample(frac=1, random_state=42).reset_index(drop=True)


def quality_to_class(q):
    # 0 = low quality  (quality <= 5)
    # 1 = medium quality (quality == 6)
    # 2 = high quality (quality >= 7)
    if q <= 5:
        return 0
    elif q == 6:
        return 1
    else:
        return 2


df["target"] = df["quality"].apply(quality_to_class)
df = df.drop(columns=["quality"])

os.makedirs("data", exist_ok=True)

n_eval = 500
n_half = (len(df) - n_eval) // 2

eval_df      = df.iloc[:n_eval]
train_phase1 = df.iloc[n_eval : n_eval + n_half]
train_phase2 = df.iloc[n_eval + n_half : n_eval + 2 * n_half]

train_phase1.to_csv("data/train_phase1.csv", index=False)
eval_df.to_csv(     "data/eval.csv",         index=False)
train_phase2.to_csv("data/train_phase2.csv", index=False)

print(f"train_phase1.csv : {len(train_phase1)} samples")
print(f"eval.csv         : {len(eval_df)} samples")
print(f"train_phase2.csv : {len(train_phase2)} samples")

for label, split_df in [("train_phase1", train_phase1), ("eval", eval_df), ("train_phase2", train_phase2)]:
    dist = split_df["target"].value_counts(normalize=True).sort_index()
    print(f"\n{label} class distribution:")
    for cls, pct in dist.items():
        name = {0: "low", 1: "medium", 2: "high"}.get(cls, "?")
        print(f"  class {cls} ({name:6s}): {pct:.2%}")

print(f"\nColumns: {list(train_phase1.columns)}")

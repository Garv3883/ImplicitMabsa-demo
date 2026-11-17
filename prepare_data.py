import pandas as pd
import os

label_map = {0: "negative", 1: "neutral", 2: "positive", "0": "negative", "1": "neutral", "2": "positive"}
rows = []

for split in ["train.tsv", "dev.tsv", "test.tsv"]:
    path = os.path.join("datasets/twitter2015", split)
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                _, label, img_id, tweet_tpl, aspect = parts[:5]
                full_text = tweet_tpl.replace("$T$", aspect.strip())
                rows.append({
                    "text": full_text,
                    "image_path": img_id.strip(),
                    "aspect": aspect.strip(),
                    "sentiment": label_map.get(label.strip(), "neutral")
                })

df = pd.DataFrame(rows)
df.to_csv("datasets/twitter2015.csv", index=False)
print(f"Done! Saved {len(df)} rows to datasets/twitter2015.csv")
print(df.head(3))

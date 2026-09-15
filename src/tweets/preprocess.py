import json
import random
import re
import pandas as pd
from pathlib import Path

BRAND = "AppleSupport"
DATA = Path("~/tweets/archive/twcs/twcs.csv")
OUT = Path("data")
OUT.mkdir(exist_ok=True)
random.seed(42)


# text cleaning
def clean(text: str, brand: str) -> str:
    text = str(text)
    text = re.sub(r"^RT @\w+:\s*", "", text)# retweet prefix
    text = re.sub(rf"@{brand}", "[BRAND]", text)
    text = re.sub(r"@\w+", "[USER]", text)             # de-identify handles
    text = re.sub(r"https?://\S+|t\.co/\S+", "[LINK]", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"\s+", " ", text).strip()


# load data
df = pd.read_csv(DATA)
if df["inbound"].dtype != bool:
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])

for col in ["tweet_id", "in_response_to_tweet_id"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Loaded {len(df):,} tweets")

# fast lookup: tweet_id -> row
tweets = df.set_index("tweet_id")


def get_row(tid):
    try:
        return tweets.loc[tid]
    except KeyError:
        return None


# threads
# A thread = start at a customer tweet, follow response_tweet_id chains.
# response_tweet_id can hold comma-separated multiple ids -> take all.

brand_replies = df[(df["author_id"] == BRAND) & (~df["inbound"])]
print(f"{BRAND} replies: {len(brand_replies):,}")

# index: parent_id -> [child tweet_ids]  (only need children of brand-replied tweets,
# but build cheap global map of id links)
children = {}
for tid, rid in zip(df["tweet_id"], df["response_tweet_id"]):
    if pd.isna(rid):
        continue
    for c in str(rid).split(","):
        c = c.strip()
        if c.isdigit():
            children.setdefault(int(tid), []).append(int(c))


def build_thread(start_id) -> dict:
    """BFS from a root customer tweet, collecting all reachable turns."""
    turns, stack = [], [start_id]
    seen = set()
    while stack:
        tid = stack.pop()
        if tid in seen:
            continue
        seen.add(tid)
        row = get_row(tid)
        if row is None:
            continue
        turns.append({
            "tweet_id": int(tid),
            "speaker": "customer" if row["inbound"] else "brand",
            "author": row["author_id"],
            "text": clean(row["text"], BRAND),
            "created_at": row["created_at"],
        })
        stack.extend(children.get(tid, []))
    turns.sort(key=lambda t: t["created_at"])
    return {"thread_id": int(start_id), "turns": turns}


# roots = customer tweets that the brand replied to (directly or in chain)
replied_ids = set()
for rid in brand_replies["in_response_to_tweet_id"].dropna().astype(int):
    row = get_row(rid)
    # walk back to the thread root
    while row is not None and not pd.isna(row["in_response_to_tweet_id"]):
        nxt = int(row["in_response_to_tweet_id"])
        if nxt in replied_ids:
            break
        row = get_row(nxt)
    if row is not None:
        replied_ids.add(int(row.name))

print(f"Threads where {BRAND} replied: {len(replied_ids):,}")

threads = [build_thread(rid) for rid in replied_ids]
threads = [t for t in threads if any(x["speaker"] == "brand" for x in t["turns"])]
print(f"Reconstructed {len(threads):,} valid threads")

# split DISJOINT by thread
random.shuffle(threads)
n_held = max(300, int(len(threads) * 0.15))
held_threads, ref_threads = threads[:n_held], threads[n_held:]


# write reference library
# one record per brand reply: customer message it answered + the reply itself
with open(OUT / "reference_library.jsonl", "w") as f:
    n = 0
    for t in ref_threads:
        turns = t["turns"]
        for i, turn in enumerate(turns):
            if turn["speaker"] == "brand" and i > 0:
                f.write(json.dumps({
                    "thread_id": t["thread_id"],
                    "customer_msg": turns[i - 1]["text"],
                    "brand_reply": turn["text"],
                    "context": [x["text"] for x in turns[:i - 1]],
                }) + "\n")
                n += 1
print(f"reference_library.jsonl: {n:,} reply pairs")


# write held-out + golden candidates
first_turns = {}
for t in held_threads:
    for turn in t["turns"]:
        if turn["speaker"] == "customer":
            first_turns.setdefault(t["thread_id"], turn)  # first seen = earliest
            break  # only the first customer turn matters

held_records = [
    {"thread_id": tid, "tweet_id": turn["tweet_id"], "text": turn["text"]}
    for tid, turn in first_turns.items()
]

golden = random.sample(held_records, min(300, len(held_records)))

with open(OUT / "held_out.jsonl", "w") as f:
    for r in held_records:
        f.write(json.dumps(r) + "\n")
with open(OUT / "golden_candidates.jsonl", "w") as f:
    for r in golden:
        f.write(json.dumps(r) + "\n")

print(f"held_out.jsonl: {len(held_records):,} first-turn messages "
      f"(from {len(held_threads):,} threads)")
print(f"golden_candidates.jsonl: {len(golden)}")

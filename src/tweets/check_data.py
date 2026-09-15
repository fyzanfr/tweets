import json

ref = [json.loads(l) for l in open("data/reference_library.jsonl")]
held = [json.loads(l) for l in open("data/held_out.jsonl")]

assert not {r["thread_id"] for r in ref} & {h["thread_id"] for h in held}, "LEAK!"
print("Leakage check passed")

print(ref[0]["customer_msg"][:120], "->", ref[0]["brand_reply"][:120])
print(len(ref), "reference pairs")
print(len(held), "held-out messages,", len({h['thread_id'] for h in held}), "threads")

import json
from collections import Counter

with open("labeled_samples.json") as f:
    raw = json.load(f)
print(Counter(role for role, _ in raw))
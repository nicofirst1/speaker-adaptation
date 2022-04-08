import json

with open('../../data/vectors.json', 'r') as f:
    vectors = json.load(f)

with open('../../data/clip.json', 'r') as f:
    clip = json.load(f)

assert len(vectors) == len(clip)

import json

domains = ["appliances", "food", "indoor", "outdoor", "vehicles", "speaker"]
splits = ["test", "train", "val"]

for d in domains:

    print(d)

    for s in splits:
        data_file = "../data/chains-domain-specific/" + d + "/" + s + ".json"

        count_utt = 0

        with open(data_file, "r") as f:
            subset = json.load(f)

            for img in subset:
                for game in subset[img]:
                    count_utt += len(subset[img][game])

            print(s, count_utt)

    print()

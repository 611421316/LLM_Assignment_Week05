import json
import os


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def loadData():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "my_project/data")

    test_data = load_jsonl(os.path.join(data_dir, "test_review_subset.json"))
    users = load_jsonl(os.path.join(data_dir, "user_subset.json"))
    items = load_jsonl(os.path.join(data_dir, "item_subset.json"))
    reviews = load_jsonl(os.path.join(data_dir, "review_subset.json"))

    pairs = [(x["user_id"], x["item_id"]) for x in test_data]

    user_reviews = {}
    for r in reviews:
        user_id = r.get("user_id")
        if user_id is None:
            continue
        if user_id not in user_reviews:
            user_reviews[user_id] = []
        user_reviews[user_id].append(r)

    item_reviews = {}
    for r in reviews:
        item_id = r.get("item_id")
        if item_id is None:
            continue
        if item_id not in item_reviews:
            item_reviews[item_id] = []
        item_reviews[item_id].append(r)

    return pairs, users, items, reviews, user_reviews, item_reviews
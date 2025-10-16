import json
import os

def save_labels(labels_dict, path="data/labels.json"):
    with open(path, 'w') as f:
        json.dump(labels_dict, f)

def load_labels(path="data/labels.json"):
    with open(path, 'r') as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
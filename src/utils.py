import csv
import json
from pathlib import Path


def load_text_file(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_csv(data, path):
    if not data:
        return
    scalar_keys = [
        "rank", "candidate_name", "file", "final_score",
        "semantic_similarity", "skill_match", "experience_match",
        "education_match", "experience_years_detected", "reasoning"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k, "") for k in scalar_keys})

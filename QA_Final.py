import json
import re
from collections import Counter
from datasets import load_dataset
from datasets import Dataset

# Normalization and Evaluation Helpers
def normalize_text(s):
    def white_space_fix(text):
        return ' '.join(text.split())
    return white_space_fix(s)

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)

# Evaluation function which will take the predicted answer and the actual answer from the dataset
def evaluate(predictions, references):
    em_scores = []
    f1_scores = []
    for id_, ref in references.items():
        pred = predictions.get(id_, "")
        em = exact_match_score(pred, ref)
        f1 = f1_score(pred, ref)
        em_scores.append(em)
        f1_scores.append(f1)
    return {
        "exact_match": 100.0 * sum(em_scores) / len(em_scores) if em_scores else 0,
        "f1": 100.0 * sum(f1_scores) / len(f1_scores) if f1_scores else 0,
    }

# Convert HuggingFace Dataset to SQuAD format for training 
def to_squad_format(hf_dataset):
    squad_data = {"data": []}

    for row in hf_dataset:
        if "data" not in row:
            continue

        for entry in row["data"]:
            if "paragraphs" not in entry:
                continue

            squad_data["data"].append({
                "title": entry.get("title", "unknown"),
                "paragraphs": entry["paragraphs"]
            })

    return squad_data

def squad_to_flat_list(squad_data):
    flat_list = []
    for article in squad_data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                flat_list.append({
                    "id": qa["id"],
                    "context": context,
                    "question": qa["question"],
                    "answers": qa["answers"]
                })
    return flat_list


def save_sample_to_file(squad_data, filename="sample_output.json", num_samples=20):
    sample_data = {
        "data": []
    }

    if len(squad_data["data"]) > 0:
        data = squad_data["data"][0]
        # Limit paragraphs to num_samples
        limited_paragraphs = data["paragraphs"][:num_samples]

        sample_data["data"].append({
            "title": data.get("title", ""),
            "paragraphs": limited_paragraphs
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"Sample saved to {filename}.")

def main():
    print(" Loading Arabic dataset from HuggingFace...")
    # Load the dataset
    dataset = load_dataset("i0xs0/Arabic-SQuAD")["train"]

    # Check if the dataset is nested
    if len(dataset) == 1 and isinstance(dataset[0], dict) and "data" in dataset[0]:
        print("Dataset is nested. Flattening it...")
        dataset = dataset[0]["data"]  # Extract the samples from the "data" key

    # Confirm the flattened dataset length
    print(f"Flattened dataset length: {len(dataset)}")

    # Convert to SQuAD format and save the flattened version
    squad_data = to_squad_format([{"data": dataset}])
    data = squad_to_flat_list(squad_data)
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Flattened SQuAD-format dataset saved to flattened_squad.json.")

    # Split the dataset into 80% train and 20% test
    dataset = Dataset.from_list(dataset)  # Convert the list of dictionaries into a HuggingFace Dataset
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    # Separate train and test datasets
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Save train and test datasets
    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    # Convert train and test datasets to SQuAD format
    train_squad = to_squad_format([{"data": train_dataset}])
    test_squad = to_squad_format([{"data": test_dataset}])

    train_flat = squad_to_flat_list(train_squad)

    # Save to JSON files
    with open("train_dataset.json", "w", encoding="utf-8") as train_file:
        json.dump(train_flat, train_file, ensure_ascii=False, indent=2)

    with open("test_dataset.json", "w", encoding="utf-8") as test_file:
        json.dump(test_squad, test_file, ensure_ascii=False, indent=2)

    print("Train and test datasets saved to train_dataset.json and test_dataset.json.")


if __name__ == "__main__":
    main()

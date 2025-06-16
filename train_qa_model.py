# train_qa_model.py
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import os
import json

# --- Configuration ---
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
DATASET_PATH = "train_dataset.json"  # Path to your flat dataset
OUTPUT_DIR = "./trained_model"
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 384
STRIDE = 128
LOGGING_DIR = "./logs"

# --- 1. Load the flat JSON dataset ---
# Assuming the JSON is a list of dictionaries like:
# [{"id": "...", "context": "...", "question": "...", "answers": {"text": [...], "answer_start": [...]}}, ...]
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    flattened_data = json.load(f)  # Already flat, no need to flatten

# Print a sample of the flattened dataset
print("Sample of flattened data:")
for i, sample in enumerate(flattened_data[:3]):  # Show the first 3 samples
    print(f"\nSample {i+1}:")
    print(f"Context: {sample['context'][:100]}...")
    print(f"Question: {sample['question']}")
    print(f"Answers: {sample['answers']}")

# Convert to Hugging Face Dataset
raw_datasets = Dataset.from_list(flattened_data)
raw_datasets = DatasetDict({"train": raw_datasets})

# --- 2: Load Tokenizer and Model ---
print(f"Loading tokenizer and model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
print("Model and tokenizer loaded successfully.")


# --- 3: Preprocess Data ---
def prepare_train_features(examples):
    # Some questions are given as lists, others as strings
    # We take the first element if it's a list.
    examples["question"] = [q[0] if isinstance(q, list) else q for q in examples["question"]]
    examples["context"] = [c[0] if isinstance(c, list) else c for c in examples["context"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        # ***** FIX: Changed truncation to "only_second" which refers to the context *****
        # The standard for QA is to truncate the context, not the question.
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # If no answers are given, set the cls_index as answer.
        if len(answers) == 0 or len(answers[0]["text"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start and end character index of the answer in the text.
            start_char = answers[0]["answer_start"]
            end_char = start_char + len(answers[0]["text"])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1: # 1 corresponds to the context
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1: # 1 corresponds to the context
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

print("Preparing training features...")
# Note: I've also swapped the order of question/context to match the new truncation strategy.
# The tokenizer expects (questions, contexts) for QA.
tokenized_datasets = raw_datasets.map(
    prepare_train_features,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
print("Features prepared successfully.")


# --- Step 4: Training Setup ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir=LOGGING_DIR,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    # Using fp16 can speed up training on compatible GPUs.
    # If you encounter issues, you can set this to False.
    fp16=True, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    # You might want to add an evaluation dataset later
    # eval_dataset=tokenized_datasets["validation"], 
    tokenizer=tokenizer,
)

# --- Step 5: Train the Model ---
print("Starting training...")
trainer.train()
print("Training finished.")

# --- Step 6: Save the Trained Model ---
print(f"Saving trained model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
# The tokenizer is already saved by the Trainer, but saving it again is fine.
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved successfully.")

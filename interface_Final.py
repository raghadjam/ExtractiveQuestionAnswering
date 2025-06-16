# qa_interface.py
import json
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from retriever_Final import TfidfRetriever
import torch
from QA_Final import evaluate,normalize_text

# Load the retriever
st.title(" سياق ")
retriever = TfidfRetriever("data.json")

with open("data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

# User input
question = st.text_input("اكتب سؤالك هنا:")

if question:
    # Retrieve top context
    top_context = retriever.retrieve(question)[0]
    st.subheader("أقرب نص من الملف:")
    st.write(top_context)

   # Find the ground truth answer from the dataset using normalization
    ground_truth = ""
    found = False
    for item in dataset:
        context_ds = item.get("context", "")
        question_ds = item.get("question", "")
        # Use normalization for robust matching
        if (
            normalize_text(context_ds) == normalize_text(top_context)
            and normalize_text(question_ds) == normalize_text(question)
        ):
            answers = item.get("answers", {})
            if isinstance(answers, dict):
                ground_truth = answers.get("text", [""])[0]
            elif isinstance(answers, list) and answers:
                ground_truth = answers[0].get("text", "")
            found = True
            break

    # If not found, try partial context match (fallback)
    if not found:
        for item in dataset:
            context_ds = item.get("context", "")
            if normalize_text(top_context) in normalize_text(context_ds):
                answers = item.get("answers", {})
                if isinstance(answers, dict):
                    ground_truth = answers.get("text", [""])[0]
                elif isinstance(answers, list) and answers:
                    ground_truth = answers[0].get("text", "")
                break

    print("User question:", question)
    print("Top context:", top_context[:100])
    print("Ground truth answer:", ground_truth)

    # Tokenize input for the model
    inputs = tokenizer(
        question,
        top_context,
        max_length=384,
        truncation="only_second",
        return_tensors="pt"
    )

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Find the answer span
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits) + 1  # End index is exclusive

    # Decode the answer
    input_ids = inputs["input_ids"][0]
    answer = tokenizer.decode(input_ids[start_index:end_index])

    # Display the answer
    st.subheader("الإجابة المستخرجة:")
    if answer.strip():
        st.success(answer)
    else:
        st.warning("❌ لم يتم العثور على إجابة واضحة.")

    predictions = {"1": answer}
    references = {"1": ground_truth}

    # Evaluate
    eval_result = evaluate(predictions, references)

    # Print to terminal
    print("Evaluation:", eval_result)

    st.subheader("الاجابة الصحيحة:")
    st.write(f"{ground_truth}")

    # Display in Streamlit
    st.subheader("تقييم الإجابة:")
    st.write(f"Exact Match: {eval_result['exact_match']:.2f}")
    st.write(f"F1 Score: {eval_result['f1']:.2f}")
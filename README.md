# Arabic Question Answering System Using AraBERT and TF-IDF Retrieval

This project is an **Arabic Question Answering (QA)** system that combines a **TF-IDF retriever** with **AraBERT**, a transformer model trained for Arabic. The system is designed to:
- Automatically retrieve the most relevant paragraph from a corpus.
- Use a fine-tuned AraBERT model to extract the answer span from the retrieved paragraph.
- Evaluate the answer using **Exact Match (EM)** and **F1 Score**.
- The system is trained and evaluated on the [**Arabic-SQuAD dataset**](https://huggingface.co/datasets/i0xs0/Arabic-SQuAD) — an Arabic translation of the Stanford QA Dataset (SQuAD), providing question-context-answer triples in Arabic.



##  Features

- **TF-IDF Retrieval**: Uses cosine similarity to find the most relevant context.
- **AraBERT Fine-Tuned Model**: Extracts answers from Arabic text using a QA model.
- **Performance Metrics**: Includes EM and F1 to measure accuracy.




##  Collaborators

- [@raghadjam](https://github.com/raghadjam)
- [@hebafialah](https://github.com/Fialah-heba)
- [@danahafithah](https://github.com/dana-hafitha)





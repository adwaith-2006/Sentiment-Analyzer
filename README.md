# Sentiment Analyzer

This project is an **AI Sentiment Analyzer** built in Python using **BERT**.  
It classifies text reviews into **Negative, Neutral, or Positive** sentiments and displays evaluation metrics and a confusion matrix.  

---

## Features

- Uses **Hugging Face Transformers** (`bert-base-uncased`) for sequence classification
- Loads and preprocesses datasets using **Datasets**
- Tokenizes text and prepares it for the model
- Computes evaluation metrics: **accuracy, F1-score, precision, recall**
- Displays a **confusion matrix** to visualize model performance
- Interactive UI (optional) with **Gradio**

---

## Requirements

- Python 3.7+
- Libraries:
  - `torch`
  - `transformers`
  - `datasets`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `gradio`
  - `evaluate`

Install all dependencies with:

```bash
pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn gradio evaluate

## How to Run

1. Open `AI_internship.ipynb` in **Colab** or **Jupyter Notebook**.

2. Run all cells sequentially.

3. The notebook will load a subset of the Yelp Review dataset, train a BERT model, and evaluate it.

4. View metrics and a confusion matrix at the end of the notebook.

5. Optionally, launch the Gradio interface if included.

## Notes

- The notebook currently **clears outputs** before pushing to GitHub to reduce file size.
- You can **rerun all cells** to regenerate the charts, metrics, and results.
- The PDF summarizer module is included but may not provide complete answers.


# AI-Generated Text Detection

An academic mini-project for distinguishing AI-generated text from human-written text. The repository contains a classical machine-learning baseline, a Streamlit application for single-text and CSV batch predictions, and experiment code and results examining detector behavior on unseen generators, false positives, and paraphrased text.

> **Important:** This detector is an experimental tool. Its prediction and confidence score are not proof of authorship and should not be used alone to make academic, employment, or other high-impact decisions.

## Project at a glance

- **Course:** IT41033 – Nature Inspired Algorithms (NIA) Mini Project
- **Task:** Binary text classification (`human` / `ai`)
- **Baseline:** TF-IDF word and bigram features with Logistic Regression
- **Interface:** Streamlit
- **Additional analysis:** Linguistic feature analysis and experiments on false positives, unseen text generators, and paraphrasing

## Features

- Enter text in the web interface to receive a predicted class and, where supported, class confidence scores.
- Upload a CSV containing a `text` column for batch predictions and download the results.
- Reuse the saved baseline model and vectorizer in `models/`.
- Explore the preprocessing, training, and analysis notebooks and scripts.
- Review saved metrics, prediction samples, and visualizations in `results/` and `outputs/`.

## Repository layout

```text
.
├── app.py                    # Streamlit prediction interface
├── models/                   # Saved model and TF-IDF vectorizer
├── src/                      # Preprocessing, feature extraction, and prediction code
├── train/                    # Baseline training script
├── notebooks/                # Data preparation, training, and experiment notebooks
├── Para_codes/               # Supporting analysis and visualization scripts
├── data/                     # Data and processed experiment inputs
├── data_p/                   # Additional project datasets and prepared training data
├── outputs/                  # Feature summaries and figures
├── results/                  # Evaluation metrics, predictions, and experiment results
└── requirements.txt          # Python dependencies
```

Dataset files may be large or have separate usage terms. Check the source and permission for each dataset before sharing or redistributing it.

## Setup

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the web app

From the repository root:

```bash
streamlit run app.py
```

The app uses `models/baseline_model.pkl` and `models/tfidf_vectorizer.pkl`. Keep both files in the `models/` directory. If either file is missing, the app will show an error.

For batch predictions, upload a CSV file with a column named `text`. The app displays the first 20 predictions and provides a CSV download with a `predicted` column.

## Train the baseline model

The training script expects the prepared dataset at `data_p/processed/dataset_comb_train.csv`, with `text`, `author`, and `split` columns. Run the script with `train/` as the working directory so its relative paths resolve correctly:

```bash
cd train
python 01_base_train.py
```

It trains a TF-IDF + Logistic Regression classifier, evaluates the test split, and saves the model, vectorizer, confusion matrix, predictions, and baseline metrics to the repository's `models/` and `results/` directories. Return to the repository root before launching the app.

## Current baseline result

The committed baseline report records **94.00% accuracy** on 13,628 test samples. The dataset is imbalanced, and the report shows notably lower recall for the human class (0.49) than for AI text (0.99). Accuracy alone therefore gives an incomplete view of performance; review the per-class precision, recall, F1-score, and confusion matrix in [`results/metrics.md`](results/metrics.md) and [`results/confusion_matrix.png`](results/confusion_matrix.png).

Other result files report separate experiments and use different evaluation sets. Compare results only with their experiment setup and sample counts in mind. In particular, a result from one held-out generator or dataset does not establish reliable performance on all text, models, or domains.

## Team

Add the five team members' preferred names and, if useful, their project roles below:

| # | Team member | Contribution / role |
|---|---|---|
| 1 | MRAR. Musharraf | Data Collection, Preprocessing, Baseline model Training |
| 2 | S. Niroshini | Generalization Testing |
| 3 | T. Devanivethitha | Feature Engineering |
| 4 | K. Theebapriya | False Positive Analysis |
| 5 | W.J. Abisha | Paraphrase Evasion Testing |

## Limitations

- Text detectors can produce both false positives and false negatives. This project itself includes analyses of those behaviors.
- Writing style, topic, text length, language proficiency, editing, and the generator or paraphrasing method can affect predictions.
- The probability-like confidence shown by the app is the classifier's score, not a guarantee that the prediction is correct.
- The saved model is a baseline and may not represent the latest result from every experiment in the repository.
- Pickle model files should only be loaded from a trusted source.

## Technologies

Python · pandas · scikit-learn · Streamlit · Pytorch · Matplotlib

## License

No license is currently specified in this repository. Contact the project team before reusing or redistributing the code or data.

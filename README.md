# Naïve Bayes Sentiment Classifier

An interactive sentiment analysis demo built **from scratch** — no scikit-learn, no NLTK, no ML libraries. Just Python, math, and vanilla JS.

Type any movie or product review and see exactly which words drive the prediction, with live log-probability breakdowns.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![NLP](https://img.shields.io/badge/NLP-from%20scratch-blueviolet)

---

## What it demonstrates

- **Bag-of-words** tokenization
- **Naïve Bayes** classification using Bayes' theorem
- **Log-probabilities** to avoid floating point underflow
- **Laplace smoothing** (α=1) so unseen words never zero out a prediction
- **Log-likelihood ratios** per word — surfaced live in the UI

---

## How it works

Naïve Bayes classifies text by computing the probability of each class given the input words:

```
log P(class | words) ∝ log P(class) + Σ log P(word | class)
```

Each word contributes independently (the "naïve" assumption). The word evidence table shows exactly how much each word pushed the prediction toward positive or negative.

---

## Project structure

```
naive-bayes-sentiment/
├── app.py                  # Flask backend, /classify endpoint
├── classifier.py           # Naïve Bayes logic (pure Python)
├── data/
│   └── training_data.py    # 50 labeled training sentences
├── static/
│   ├── style.css           # Dark-themed UI
│   └── script.js           # Fetch, render results, evidence table
├── templates/
│   └── index.html          # Main demo page
└── requirements.txt
```

---

## Running locally

```bash
git clone git@github.com:derrickliu/naive-bayes-sentiment.git
cd naive-bayes-sentiment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 app.py
```

Then open http://127.0.0.1:5000

---

## Example output

Input: *"this movie was absolutely wonderful"*

| Word | Diff |
|------|------|
| wonderful | +0.6904 |
| absolutely | +0.6904 |
| movie | −0.0028 |
| this | −0.2905 |
| was | +0.6904 |

Predicted: **positive** (85.5% confidence)
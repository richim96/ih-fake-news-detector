# Fake News Detector

A simple NLP pipeline for classifying news headlines as **real (1)** or **fake (0)**.

## Project Structure

```
ih-fake-news-detector/
├── main.ipynb
├── data/
│   ├── raw/
│   │   ├── training_data.csv
│   │   └── testing_data.csv
│   ├── processed/
│   │   └── testing_data_predicted.csv
└── README.md
```

## Installation
```bash
# Install dependencies
uv sync
```

## Dataset

| Split | Samples | Fake (0) | Real (1) |
|-------|---------|----------|----------|
| Train | 34,152  | ~51.5%   | ~48.5%   |
| Test  | 9,984   | —        | —        |


## Pipeline

### 1. Text Preprocessing

- Fix mojibake characters (`‚î`, `√©`, etc.)
- Normalize country abbreviations (`u.s.` → `usa`)
- Expand contractions (`don't` → `do not`)
- Strip non-alphanumeric characters
- Remove English stopwords
- Lemmatize tokens (WordNetLemmatizer)

### 2. Feature Extraction

| Method    | Dimensions | Notes |
|-----------|-----------|-------|
| BoW       | 14,931    | CountVectorizer, fit on train only |
| TF-IDF    | 14,931    | TfidfVectorizer, fit on train only |
| Word2Vec  | 100       | Trained from scratch on train corpus; mean-pooled |
| fastText  | 300       | Pretrained `cc.en.300.bin`; subword-aware |
| BERT      | 768       | `bert-base-uncased` CLS token; no fine-tuning |

### 3. Classifiers

- **Logistic Regression** — linear baseline (`max_iter=1000`)
- **Random Forest** — 200 trees, all CPU cores
- **Naive Bayes** — MultinomialNB for sparse (BoW/TF-IDF), GaussianNB for dense

> Gradient Boosting was tested but dropped due to poor performance and slow training.

## Results (Validation Set — Macro F1)

| Rank | Feature   | Classifier          | Macro F1 |
|------|-----------|---------------------|----------|
| 1    | TF-IDF    | Logistic Regression | **0.9349** |
| 2    | BoW       | Logistic Regression | 0.9348   |
| 3    | BoW       | Naive Bayes         | 0.9335   |
| 4    | TF-IDF    | Naive Bayes         | 0.9301   |
| 5    | TF-IDF    | Random Forest       | 0.9196   |
| 6    | BoW       | Random Forest       | 0.9151   |
| 7    | Word2Vec  | Random Forest       | 0.9097   |
| 8    | fastText  | Logistic Regression | 0.9029   |
| 9    | BERT      | Logistic Regression | 0.9018   |
| 10   | Word2Vec  | Logistic Regression | 0.8981   |
| 11   | fastText  | Random Forest       | 0.8951   |
| 12   | BERT      | Random Forest       | 0.8805   |
| 13   | Word2Vec  | Naive Bayes         | 0.8628   |
| 14   | fastText  | Naive Bayes         | 0.8586   |
| 15   | BERT      | Naive Bayes         | **0.8101** |

### Key Findings

- **Sparse bag-of-words methods (BoW, TF-IDF) dominate** — simple token frequency is highly informative for headline-level fake news detection.
- **Logistic Regression is the best classifier** across all feature types, benefiting from the high-dimensional sparse representations.
- **BERT underperforms** classical methods without fine-tuning — the frozen CLS embedding alone is not sufficient.
- **Top-4 models all exceed 0.93 F1**, showing strong, consistent performance from classical NLP.

## Test Predictions

The best model (TF-IDF + Logistic Regression) was used to generate predictions:

- **Output:** `data/processed/testing_data_predicted.csv`
- **Format:** same as input (`is_real\ttext`, no header)
- **Distribution:** 4,849 fake (0) / 5,135 real (1)



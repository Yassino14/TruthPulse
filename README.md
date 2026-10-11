# TruthPulse: AI Misinformation & Social Media Sentiment Analysis

A machine learning university project developed for **4 DSIA**, featuring:
1. **Fake News & Misinformation Taxonomy Detector**
2. **Social Media Sentiment Analysis**
3. **Full ETL Data Engineering Pipeline**
4. **Interactive Streamlit Web Dashboard**

---

## 🌟 Key Highlights & Results

- **Sentiment Analysis Model**:
  - Dataset: 1.6 Million raw tweets from **Sentiment140**
  - Preprocessing: Unescape HTML, mention stripping, URL removal, character elongation normalization, stratified sampling (80,000 tweets)
  - Model: TF-IDF (30,000 unigrams + bigrams) + Logistic Regression
  - **Accuracy: 78.89%** | **F1-Score: 78.92%** | **Inference: < 10 ms**

- **Fake News & Misinformation Classifier**:
  - Dataset: 12,999 raw articles with 20 attributes (`fake.csv`)
  - Preprocessing: Language filtering, missing value imputation, boilerplate removal, feature fusion (`title + text`)
  - Models:
    - **Binary Fabrication Detector**: **88.68% Accuracy** | **89.15% Weighted F1**
    - **Misinformation Taxonomy Classifier**: **85.02% Accuracy** (Categories: *Fabricated/BS, Conspiracy Theory, Extreme Bias, Hate Speech, Satire, State Propaganda*)

---

## 🏗️ Project Architecture

```
Projet ML/
├── data/
│   ├── raw/
│   │   ├── fake.csv                             # Raw 12,999 fake news articles (54 MB)
│   │   └── sentiment140.csv                     # Raw 1.6M tweets (238 MB)
│   └── processed/
│       ├── fake_clean.csv                       # Clean, normalized article corpus (12,011 rows)
│       └── sentiment_clean.csv                  # Clean, balanced tweets (80,000 rows)
├── models/
│   ├── fake_bin_model.joblib                    # Binary fabrication classifier
│   ├── fake_bin_vectorizer.joblib               # TF-IDF vectorizer (25k features)
│   ├── fake_multi_model.joblib                  # Multi-class taxonomy classifier
│   ├── fake_multi_vectorizer.joblib             # Multi-class TF-IDF vectorizer
│   ├── sentiment_model.joblib                   # Sentiment logistic regression
│   ├── sentiment_vectorizer.joblib              # Sentiment TF-IDF vectorizer (30k ngrams)
│   ├── metrics.json                             # Evaluation metrics & confusion matrices
│   └── etl_report.json                          # Audit statistics from the ETL pipeline
├── src/
│   ├── etl.py                                   # Full Extract, Transform, Load pipeline
│   ├── train.py                                 # Training, evaluation & artifact persistence
│   ├── predictor.py                             # Reusable inference engine with explainability
│   └── analyze_datasets.py                      # Exploratory data profiling script
├── app.py                                       # Interactive Streamlit Web Dashboard
├── requirements.txt                             # Python dependencies
└── README.md                                    # Project documentation
```

---

## ⚙️ Data Engineering & ETL Pipeline (`src/etl.py`)

### 1. Social Media Sentiment ETL:
* **Extract:** Ingests `sentiment140.csv` (1.6M rows) without native headers. Assigns canonical schema `[target, id, date, flag, user, text]`.
* **Transform:**
  * Re-encodes target from `[0, 4]` to `0: Negative` and `1: Positive`.
  * Removes 18,534 duplicate tweet texts.
  * Cleans noise via regular expressions: strips `@mentions`, raw URLs, HTML entities (`&quot;`, `&amp;`), and normalizes elongated strings (e.g. `looooove` -> `loove`).
  * Filters out empty/stub strings.
  * Applies **balanced stratified sampling** to 80,000 high-signal records.
* **Load:** Exports to `data/processed/sentiment_clean.csv`.

### 2. Fake News ETL:
* **Extract:** Ingests `fake.csv` (12,999 rows, 20 columns).
* **Transform:**
  * Filters non-English records (596 dropped).
  * Imputes missing authors and missing headlines.
  * Drops invalid records (< 30 characters).
  * Harmonizes raw types into taxonomy categories (*Fabricated / BS, Conspiracy Theory, Extreme Bias, Hate Speech, Satire, State Propaganda*).
  * Constructs full-text composite representation (`title + text`).
  * Strips scraping artifacts, HTML tags, and trailing advertising boilerplates.
  * Removes 148 duplicate articles.
* **Load:** Exports to `data/processed/fake_clean.csv`.

---

## 🚀 How to Run the Streamlit Application

In your terminal / PowerShell:

```powershell
# Navigate to the project directory
cd "c:\Users\louat\Desktop\4 DSIA\Projet ML"

# Launch Streamlit
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Features

1. **📰 Fake News Detector Tab:**
   - Real-time headline & body analysis.
   - Quick test buttons for immediate demonstration.
   - Fabrication probability meter & category prediction.
   - Token-level explainability triggers.

2. **💬 Social Media Sentiment Tab:**
   - Real-time tweet & post analysis.
   - Positive vs. Negative probability distribution.
   - Shows ETL-cleaned text comparison.
   - Word importance table extracted from model coefficients.

3. **📁 Batch CSV Predictor Tab:**
   - Upload any `.csv` with text columns.
   - Run bulk predictions on hundreds of rows simultaneously.
   - Download the annotated results as a CSV file.

4. **📊 Model Performance & Metrics Tab:**
   - Interactive Confusion Matrices (Heatmaps).
   - Accuracy, Precision, Recall, and F1-Scores.

5. **⚙️ ETL Engineering Pipeline Tab:**
   - Audit trail of the data engineering steps.
   - Before vs. after dataset statistics.
   - Misinformation category distribution charts.

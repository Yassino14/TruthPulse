import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def train_sentiment_model(clean_path: str, models_dir: str) -> dict:
    print("\n" + "="*50)
    print("TRAINING MODEL 1: SOCIAL MEDIA SENTIMENT ANALYSIS")
    print("="*50)
    
    df = pd.read_csv(clean_path)
    print(f"Loaded {len(df):,} cleaned sentiment records.")
    
    X = df['clean_text'].fillna('').astype(str).tolist()
    y = np.array(df['label'].values, dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")
    
    print("Fitting TF-IDF Vectorizer (unigrams + bigrams, max_features=30,000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression classifier (C=2.0)...")
    model = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"\n[SENTIMENT RESULTS]")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("\nConfusion Matrix:\n", cm)
    
    # Extract top words
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    top_pos_idx = np.argsort(coef)[-15:][::-1]
    top_neg_idx = np.argsort(coef)[:15]
    
    top_positive_words = [feature_names[i] for i in top_pos_idx]
    top_negative_words = [feature_names[i] for i in top_neg_idx]
    
    # Save artifacts
    model_path = os.path.join(models_dir, "sentiment_model.joblib")
    vec_path = os.path.join(models_dir, "sentiment_vectorizer.joblib")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Saved sentiment artifacts to {models_dir}")
    
    return {
        "model_name": "Logistic Regression + TF-IDF (1-2 ngrams)",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "labels": ["Negative", "Positive"],
        "top_positive_words": top_positive_words,
        "top_negative_words": top_negative_words,
        "test_size": len(y_test)
    }

def train_fake_news_models(clean_path: str, models_dir: str) -> dict:
    print("\n" + "="*50)
    print("TRAINING MODEL 2: FAKE NEWS & MISINFORMATION CLASSIFIER")
    print("="*50)
    
    df = pd.read_csv(clean_path)
    print(f"Loaded {len(df):,} cleaned fake news records.")
    
    # 1. Binary Classification: Blatant Fabricated/BS (1) vs Other Misinformation/Conspiracy/Bias (0)
    X = df['clean_content'].fillna('').astype(str).tolist()
    y_bin = np.array(df['is_fabricated_bs'].values, dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    print("Fitting TF-IDF Vectorizer on news articles (max_features=25,000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        min_df=3,
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Balanced Logistic Regression for Fabrication Detection...")
    bin_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    bin_model.fit(X_train_vec, y_train)
    
    y_pred_bin = bin_model.predict(X_test_vec)
    acc_bin = accuracy_score(y_test, y_pred_bin)
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(y_test, y_pred_bin, average='weighted')
    cm_bin = confusion_matrix(y_test, y_pred_bin).tolist()
    
    print(f"[BINARY FABRICATION RESULTS]")
    print(f"Accuracy:  {acc_bin*100:.2f}%")
    print(f"Weighted F1: {f1_bin*100:.2f}%")
    
    # 2. Multi-Class Taxonomy Classification
    # Group categories with > 50 samples
    cat_counts = df['misinfo_category'].value_counts()
    valid_cats = cat_counts[cat_counts >= 100].index.tolist()
    df_multi = df[df['misinfo_category'].isin(valid_cats)].copy()
    
    X_m = df_multi['clean_content'].fillna('').astype(str).tolist()
    y_m = np.array(df_multi['misinfo_category'].values, dtype=str)
    
    X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(
        X_m, y_m, test_size=0.2, random_state=42, stratify=y_m
    )
    
    multi_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=3,
        stop_words='english'
    )
    X_m_tr_v = multi_vec.fit_transform(X_m_tr)
    X_m_te_v = multi_vec.transform(X_m_te)
    
    print("Training Multi-Class Classifier for Misinformation Categories...")
    multi_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    multi_model.fit(X_m_tr_v, y_m_tr)
    
    y_m_pred = multi_model.predict(X_m_te_v)
    acc_multi = accuracy_score(y_m_te, y_m_pred)
    prec_m, rec_m, f1_multi, _ = precision_recall_fscore_support(y_m_te, y_m_pred, average='weighted')
    cm_multi = confusion_matrix(y_m_te, y_m_pred, labels=multi_model.classes_).tolist()
    
    print(f"[TAXONOMY CLASSIFICATION RESULTS]")
    print(f"Accuracy:  {acc_multi*100:.2f}%")
    print(f"Weighted F1: {f1_multi*100:.2f}%")
    
    # Top features for binary model
    feature_names = vectorizer.get_feature_names_out()
    coef = bin_model.coef_[0]
    top_bs_idx = np.argsort(coef)[-15:][::-1]
    top_other_idx = np.argsort(coef)[:15]
    
    top_fabricated_words = [feature_names[i] for i in top_bs_idx]
    top_contextual_words = [feature_names[i] for i in top_other_idx]
    
    # Save artifacts
    joblib.dump(bin_model, os.path.join(models_dir, "fake_bin_model.joblib"))
    joblib.dump(vectorizer, os.path.join(models_dir, "fake_bin_vectorizer.joblib"))
    joblib.dump(multi_model, os.path.join(models_dir, "fake_multi_model.joblib"))
    joblib.dump(multi_vec, os.path.join(models_dir, "fake_multi_vectorizer.joblib"))
    print(f"Saved fake news artifacts to {models_dir}")
    
    return {
        "binary_detector": {
            "model_name": "Balanced Logistic Regression + TF-IDF",
            "accuracy": round(float(acc_bin), 4),
            "precision": round(float(prec_bin), 4),
            "recall": round(float(rec_bin), 4),
            "f1_score": round(float(f1_bin), 4),
            "confusion_matrix": cm_bin,
            "labels": ["Opinion / Bias / Satire / Conspiracy", "Fabricated / Blatant BS"],
            "top_fabricated_indicators": top_fabricated_words,
            "top_other_indicators": top_contextual_words
        },
        "taxonomy_classifier": {
            "model_name": "Multi-Class Logistic Regression",
            "accuracy": round(float(acc_multi), 4),
            "weighted_f1": round(float(f1_multi), 4),
            "classes": list(multi_model.classes_),
            "confusion_matrix": cm_multi
        }
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    clean_sentiment = os.path.join(base_dir, "data", "processed", "sentiment_clean.csv")
    clean_fake = os.path.join(base_dir, "data", "processed", "fake_clean.csv")
    
    metrics = {}
    if os.path.exists(clean_sentiment):
        metrics['sentiment'] = train_sentiment_model(clean_sentiment, models_dir)
        
    if os.path.exists(clean_fake):
        metrics['fake_news'] = train_fake_news_models(clean_fake, models_dir)
        
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n[COMPLETE] All ML models trained and evaluation saved to {metrics_path}")

if __name__ == "__main__":
    main()

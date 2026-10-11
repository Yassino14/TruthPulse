import os
import re
import html
import json
import pandas as pd
import numpy as np

def clean_tweet_text(text: str) -> str:
    """ETL Transformation for raw social media / tweet text."""
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities (&amp;, &quot;, &lt;, etc.)
    text = html.unescape(text)
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove @mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # 4. Normalize elongated characters (e.g., 'sooooo goooood' -> 'soo good')
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
    
    # 5. Remove leftover unwanted characters, keeping basic punctuation for sentiment (! and ?)
    text = re.sub(r'[^\w\s!?.,\'-]', ' ', text)
    
    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_article_text(text: str) -> str:
    """ETL Transformation for raw news article text."""
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML
    text = html.unescape(text)
    
    # 2. Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Remove web-scraping boilerplates
    boilerplate_patterns = [
        r'Add To The Conversation Using Facebook Comments',
        r'Share on Facebook',
        r'Share on Twitter',
        r'Advertisement',
        r'CLICK HERE TO SUBSCRIBE',
        r'Follow us on \w+'
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
    # 4. Remove raw URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_sentiment_etl(raw_path: str, clean_path: str, sample_size: int = 80000) -> dict:
    """
    Extract, Transform, and Load the raw Sentiment140 dataset.
    """
    print("\n" + "="*50)
    print("STARTING ETL: SOCIAL MEDIA SENTIMENT (Sentiment140)")
    print("="*50)
    
    # --- EXTRACT ---
    print(f"[EXTRACT] Reading raw data from: {raw_path}")
    raw_columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df_raw = pd.read_csv(raw_path, encoding='latin-1', header=None, names=raw_columns)
    initial_rows = len(df_raw)
    print(f"[EXTRACT] Successfully extracted {initial_rows:,} raw records.")
    
    # --- TRANSFORM ---
    print("[TRANSFORM] Dropping non-essential tracking metadata (id, flag, user)...")
    df = df_raw[['target', 'date', 'text']].copy()
    
    print("[TRANSFORM] Encoding target labels (0 -> Negative, 4 -> Positive)...")
    df['sentiment'] = df['target'].map({0: 'Negative', 4: 'Positive'})
    df['label'] = df['target'].map({0: 0, 4: 1})
    
    print("[TRANSFORM] Deduplicating tweet texts...")
    dups_count = df['text'].duplicated().sum()
    df = df.drop_duplicates(subset=['text']).copy()
    
    print(f"[TRANSFORM] Cleaning text (removing mentions, URLs, HTML entities, normalizing elongated letters)...")
    df['clean_text'] = df['text'].apply(clean_tweet_text)
    
    # Filter empty or trivial tweets
    df = df[df['clean_text'].str.len() >= 3].copy()
    
    print(f"[TRANSFORM] Performing stratified sampling to {sample_size:,} balanced rows...")
    # Balanced stratified sample
    df_pos = df[df['label'] == 1].sample(n=sample_size // 2, random_state=42)
    df_neg = df[df['label'] == 0].sample(n=sample_size // 2, random_state=42)
    df_sampled = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Feature engineering: character & word length
    df_sampled['char_count'] = df_sampled['clean_text'].str.len()
    df_sampled['word_count'] = df_sampled['clean_text'].str.split().str.len()
    
    # --- LOAD ---
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    print(f"[LOAD] Saving clean dataset to: {clean_path}")
    df_sampled.to_csv(clean_path, index=False)
    
    summary = {
        "dataset": "Social Media Sentiment",
        "initial_rows": initial_rows,
        "duplicates_removed": int(dups_count),
        "processed_rows": len(df_sampled),
        "positive_count": int((df_sampled['label'] == 1).sum()),
        "negative_count": int((df_sampled['label'] == 0).sum()),
        "avg_words_per_tweet": float(round(df_sampled['word_count'].mean(), 2))
    }
    print("[ETL COMPLETE] Summary:", summary)
    return summary

def run_fake_news_etl(raw_path: str, clean_path: str) -> dict:
    """
    Extract, Transform, and Load the raw Fake News dataset.
    """
    print("\n" + "="*50)
    print("STARTING ETL: FAKE NEWS & MISINFORMATION DATASET")
    print("="*50)
    
    # --- EXTRACT ---
    print(f"[EXTRACT] Reading raw data from: {raw_path}")
    df_raw = pd.read_csv(raw_path, encoding='utf-8')
    initial_rows = len(df_raw)
    print(f"[EXTRACT] Successfully extracted {initial_rows:,} raw records with {len(df_raw.columns)} attributes.")
    
    # --- TRANSFORM ---
    print("[TRANSFORM] Filtering by language (keeping English articles)...")
    df = df_raw[df_raw['language'] == 'english'].copy()
    non_english_dropped = initial_rows - len(df)
    
    print("[TRANSFORM] Imputing missing values (title, author, spam_score)...")
    df['title'] = df['title'].fillna('').astype(str)
    df['author'] = df['author'].fillna('Unknown').astype(str)
    df['text'] = df['text'].fillna('').astype(str)
    df['spam_score'] = pd.to_numeric(df['spam_score'], errors='coerce').fillna(0.0)
    
    print("[TRANSFORM] Dropping records with empty or stub text (< 30 characters)...")
    df = df[df['text'].str.strip().str.len() >= 30].copy()
    
    print("[TRANSFORM] Mapping misinformation categories...")
    # Harmonizing types into academic misinformation taxonomies
    category_map = {
        'bs': 'Fabricated / BS',
        'fake': 'Fabricated / BS',
        'junksci': 'Junk Science',
        'conspiracy': 'Conspiracy Theory',
        'state': 'State Propaganda',
        'bias': 'Extreme Bias',
        'hate': 'Hate Speech',
        'satire': 'Satire'
    }
    df['misinfo_category'] = df['type'].map(category_map).fillna('Other Misinformation')
    
    # Binary classification: Blatant Fake/BS (1) vs Other Misinformation/Conspiracy/Bias (0)
    # This provides a clean binary prediction target
    df['is_fabricated_bs'] = df['type'].apply(lambda t: 1 if t in ['bs', 'fake', 'junksci'] else 0)
    
    print("[TRANSFORM] Constructing composite full-text feature (Title + Text)...")
    df['raw_full_content'] = df['title'] + " " + df['text']
    df['clean_content'] = df['raw_full_content'].apply(clean_article_text)
    
    print("[TRANSFORM] Removing duplicates on clean text...")
    dups_count = df['clean_content'].duplicated().sum()
    df = df.drop_duplicates(subset=['clean_content']).copy()
    
    # Metrics
    df['word_count'] = df['clean_content'].str.split().str.len()
    
    # Keep essential cleaned columns
    keep_cols = [
        'uuid', 'author', 'site_url', 'country', 'spam_score',
        'type', 'misinfo_category', 'is_fabricated_bs',
        'title', 'clean_content', 'word_count'
    ]
    df_clean = df[[c for c in keep_cols if c in df.columns]].copy()
    
    # --- LOAD ---
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    print(f"[LOAD] Saving clean dataset to: {clean_path}")
    df_clean.to_csv(clean_path, index=False)
    
    summary = {
        "dataset": "Fake News & Misinformation",
        "initial_rows": initial_rows,
        "non_english_dropped": int(non_english_dropped),
        "duplicates_removed": int(dups_count),
        "clean_rows": len(df_clean),
        "categories_distribution": df_clean['misinfo_category'].value_counts().to_dict(),
        "avg_words_per_article": float(round(df_clean['word_count'].mean(), 2))
    }
    print("[ETL COMPLETE] Summary:", summary)
    return summary

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_sentiment = os.path.join(base_dir, "data", "raw", "sentiment140.csv")
    clean_sentiment = os.path.join(base_dir, "data", "processed", "sentiment_clean.csv")
    
    raw_fake = os.path.join(base_dir, "data", "raw", "fake.csv")
    clean_fake = os.path.join(base_dir, "data", "processed", "fake_clean.csv")
    
    etl_stats = {}
    
    if os.path.exists(raw_sentiment):
        etl_stats['sentiment'] = run_sentiment_etl(raw_sentiment, clean_sentiment, sample_size=80000)
    else:
        print(f"Error: Raw sentiment file not found at {raw_sentiment}")
        
    if os.path.exists(raw_fake):
        etl_stats['fake_news'] = run_fake_news_etl(raw_fake, clean_fake)
    else:
        print(f"Error: Raw fake news file not found at {raw_fake}")
        
    # Save ETL report for Streamlit dashboard display
    report_path = os.path.join(base_dir, "models", "etl_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(etl_stats, f, indent=4)
    print(f"\n[REPORT] Saved ETL audit report to {report_path}")

if __name__ == "__main__":
    main()

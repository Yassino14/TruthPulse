import pandas as pd
import numpy as np
import os
import re

def analyze_csv_header(file_path, n_lines=5):
    print(f"--- FIRST {n_lines} RAW LINES OF {os.path.basename(file_path)} ---")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i in range(n_lines):
                line = f.readline()
                if not line:
                    break
                print(f"Line {i+1}: {line.strip()[:150]}...")
    except Exception as e:
        print(f"Error reading raw lines: {e}")
    print()

def analyze_fake_news(file_path):
    print("=" * 60)
    print("ANALYSIS: FAKE NEWS DATASET")
    print("=" * 60)
    
    # Try reading with utf-8 or latin-1
    for enc in ['utf-8', 'latin-1', 'ISO-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=enc, nrows=10000)
            print(f"Successfully read sample with encoding: {enc}")
            encoding = enc
            break
        except Exception:
            continue
            
    df_full = pd.read_csv(file_path, encoding=encoding)
    print(f"Total Rows: {len(df_full):,}")
    print(f"Total Columns: {len(df_full.columns)}")
    print(f"Column Names: {list(df_full.columns)}")
    print(f"Data Types:\n{df_full.dtypes}\n")
    
    print("--- Missing Values ---")
    null_counts = df_full.isnull().sum()
    null_pct = (null_counts / len(df_full)) * 100
    missing_df = pd.DataFrame({'Missing Count': null_counts, 'Missing Pct (%)': null_pct})
    print(missing_df)
    
    print("\n--- Duplicates ---")
    print(f"Duplicate rows count: {df_full.duplicated().sum():,}")
    
    # Check if there is a label column or what columns exist
    for col in df_full.columns:
        print(f"\nValue counts for column '{col}' (Top 5 / unique count: {df_full[col].nunique()}):")
        print(df_full[col].value_counts(dropna=False).head(5))
        
    # Text characteristics on text/title
    for text_col in ['text', 'title', df_full.columns[0]]:
        if text_col in df_full.columns and df_full[text_col].dtype == 'object':
            sample_series = df_full[text_col].dropna().astype(str)
            lengths = sample_series.str.len()
            words = sample_series.str.split().str.len()
            has_urls = sample_series.str.contains(r'https?://|www\.', regex=True).sum()
            has_html = sample_series.str.contains(r'<[^>]+>|&[a-z]+;', regex=True).sum()
            print(f"\n--- Text Analysis on column '{text_col}' ---")
            print(f"Character Length: Mean={lengths.mean():.1f}, Median={lengths.median():.1f}, Min={lengths.min()}, Max={lengths.max()}")
            print(f"Word Count: Mean={words.mean():.1f}, Median={words.median():.1f}, Min={words.min()}, Max={words.max()}")
            print(f"Contains URLs: {has_urls:,} ({has_urls/len(sample_series)*100:.2f}%)")
            print(f"Contains HTML/Entities: {has_html:,} ({has_html/len(sample_series)*100:.2f}%)")
            break

def analyze_sentiment(file_path):
    print("\n" + "=" * 60)
    print("ANALYSIS: SENTIMENT DATASET (Sentiment140)")
    print("=" * 60)
    
    # Check header vs no header
    # Standard Sentiment140 has no header and 6 columns: target, id, date, flag, user, text
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    try:
        # Load sample first to test
        df_sample = pd.read_csv(file_path, encoding='latin-1', header=None, nrows=10000)
        print(f"Sample read success. Detected {df_sample.shape[1]} columns.")
        
        # Now read full file
        df = pd.read_csv(file_path, encoding='latin-1', header=None, names=columns)
        print(f"Total Rows: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
        
        print("\n--- Missing Values ---")
        null_counts = df.isnull().sum()
        print(null_counts)
        
        print("\n--- Duplicates ---")
        print(f"Duplicate rows: {df.duplicated().sum():,}")
        print(f"Duplicate tweet texts: {df['text'].duplicated().sum():,}")
        
        print("\n--- Target Distribution ---")
        target_counts = df['target'].value_counts()
        print(target_counts)
        print("Target Percentages:")
        print(df['target'].value_counts(normalize=True) * 100)
        
        print("\n--- Unique Users ---")
        print(f"Number of unique users: {df['user'].nunique():,}")
        
        print("\n--- Date Range Sample ---")
        print(f"Sample dates:\n{df['date'].head(3).tolist()}")
        
        # Social media noise detection
        sample_tweets = df['text'].sample(20000, random_state=42).astype(str)
        has_mentions = sample_tweets.str.contains(r'@\w+', regex=True).sum()
        has_urls = sample_tweets.str.contains(r'https?://|www\.', regex=True).sum()
        has_html_entities = sample_tweets.str.contains(r'&[a-z0-9]+;', regex=True).sum()
        repeat_pat = re.compile(r'([a-zA-Z])\1{2,}')
        has_repeated_chars = sample_tweets.apply(lambda s: bool(repeat_pat.search(s))).sum()
        
        print("\n--- Social Media Artifacts in Sample (20,000 tweets) ---")
        print(f"Contains @mentions: {has_mentions:,} ({has_mentions/200:.2f}%)")
        print(f"Contains URLs: {has_urls:,} ({has_urls/200:.2f}%)")
        print(f"Contains HTML entities (&quot;, &amp;, etc.): {has_html_entities:,} ({has_html_entities/200:.2f}%)")
        print(f"Contains repeated characters (e.g. looooove): {has_repeated_chars:,} ({has_repeated_chars/200:.2f}%)")
        
        tweet_lengths = sample_tweets.str.len()
        tweet_words = sample_tweets.str.split().str.len()
        print("\n--- Tweet Length Distribution ---")
        print(f"Char Length: Mean={tweet_lengths.mean():.1f}, Median={tweet_lengths.median():.1f}, Min={tweet_lengths.min()}, Max={tweet_lengths.max()}")
        print(f"Word Count: Mean={tweet_words.mean():.1f}, Median={tweet_words.median():.1f}, Min={tweet_words.min()}, Max={tweet_words.max()}")
        
        print("\n--- Sample Raw Tweets ---")
        for i, (tgt, t) in enumerate(zip(df['target'].head(3), df['text'].head(3))):
            print(f"[{'Positive' if tgt == 4 else 'Negative'}] {t}")

    except Exception as e:
        print(f"Error analyzing sentiment dataset: {e}")

if __name__ == "__main__":
    fake_path = "fake.csv"
    sentiment_path = "training.1600000.processed.noemoticon.csv"
    
    if os.path.exists(fake_path):
        analyze_csv_header(fake_path)
        analyze_fake_news(fake_path)
    else:
        print(f"File not found: {fake_path}")
        
    if os.path.exists(sentiment_path):
        analyze_csv_header(sentiment_path)
        analyze_sentiment(sentiment_path)
    else:
        print(f"File not found: {sentiment_path}")

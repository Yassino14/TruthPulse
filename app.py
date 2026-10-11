import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.predictor import ModelPredictor

# Page Configuration
st.set_page_config(
    page_title="TruthPulse | NLP & Misinformation Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphic cards, modern badges, typography)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2B3D 50%, #3B3355 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .main-header h1 {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #A78BFA, #F472B6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
        margin-bottom: 0.4rem;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #F3F4F6;
    }
    
    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-pos {
        background-color: #065F46;
        color: #34D399;
        border: 1px solid #059669;
    }
    .badge-neg {
        background-color: #881337;
        color: #FB7185;
        border: 1px solid #E11D48;
    }
    .badge-fake {
        background-color: #7F1D1D;
        color: #F87171;
        border: 1px solid #DC2626;
    }
    .badge-other {
        background-color: #1E3A8A;
        color: #93C5FD;
        border: 1px solid #2563EB;
    }
    
    .card-box {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Predictor and Metrics Cached
@st.cache_resource
def get_predictor():
    return ModelPredictor()

@st.cache_data
def load_metrics_and_etl():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "models", "metrics.json")
    etl_path = os.path.join(base_dir, "models", "etl_report.json")
    
    metrics, etl_report = {}, {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(etl_path):
        with open(etl_path, "r", encoding="utf-8") as f:
            etl_report = json.load(f)
    return metrics, etl_report

predictor = get_predictor()
metrics, etl_report = load_metrics_and_etl()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.title("TruthPulse AI")
    
    st.markdown("---")
    st.markdown("### 📌 Active ML Pipelines")
    st.info("**1. Fake News Classifier**\n- Model: Logistic Regression\n- Vectorizer: TF-IDF (25k features)\n- Source: 12,999 Raw News Articles")
    st.success("**2. Sentiment Analyzer**\n- Model: Logistic Regression\n- Vectorizer: TF-IDF (30k ngrams)\n- Source: 1.6M Sentiment140 Tweets")
    
    st.markdown("---")
    st.caption("Developed with Python 3.14, Scikit-Learn & Streamlit")

# Main Header Banner
st.markdown("""
<div class="main-header">
    <h1>TruthPulse: Misinformation & Social Sentiment AI</h1>
    <p style="margin: 0; font-size: 1.05rem; opacity: 0.85;">
        An end-to-end Machine Learning intelligence dashboard powered by custom ETL pipelines, 
        dual TF-IDF models, and real-time natural language inference.
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    sent_acc = metrics.get('sentiment', {}).get('accuracy', 0.7889) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sentiment Accuracy</div>
        <div class="metric-val" style="color: #34D399;">{sent_acc:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    fake_acc = metrics.get('fake_news', {}).get('binary_detector', {}).get('accuracy', 0.8868) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Fake News Detection</div>
        <div class="metric-val" style="color: #60A5FA;">{fake_acc:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    raw_vol = (etl_report.get('sentiment', {}).get('initial_rows', 1600000) + 
               etl_report.get('fake_news', {}).get('initial_rows', 12999))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Raw Records Ingested</div>
        <div class="metric-val" style="color: #F472B6;">{raw_vol:,}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Inference Speed</div>
        <div class="metric-val" style="color: #A78BFA;">&lt; 15 ms</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tab_fake, tab_sent, tab_batch, tab_models, tab_etl = st.tabs([
    "📰 Fake News Detector",
    "💬 Social Media Sentiment",
    "📁 Batch CSV Predictor",
    "📊 Model Performance & Metrics",
    "⚙️ ETL Engineering Pipeline"
])

# ==========================================
# TAB 1: FAKE NEWS DETECTOR
# ==========================================
with tab_fake:
    st.subheader("📰 Misinformation & Fake News Detector")
    st.write("Input a news headline and article body to evaluate authenticity, flag fabrications, and classify misinformation type.")
    
    # Pre-loaded sample selector
    st.markdown("**💡 Quick Test Samples:**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    selected_sample = None
    if sample_col1.button("🚨 Fabricated Conspiracy", use_container_width=True):
        selected_sample = {
            "title": "Secret Deep State Labs Caught Manipulating Weather With Giant Satellites",
            "text": "Whistleblower documents leaked from covert bases prove that high-frequency atmospheric transmitters are deliberately creating massive storms to trigger emergency lockdowns and force citizens inside."
        }
    if sample_col2.button("⚖️ Highly Partisan Commentary", use_container_width=True):
        selected_sample = {
            "title": "Corrupt Bureaucrats Scheme To Destroy Independent Economy",
            "text": "The latest government regulatory report demonstrates a deliberate bias aimed at destroying traditional family enterprises while subsidizing elite international conglomerates."
        }
    if sample_col3.button("🎭 Satirical Parody News", use_container_width=True):
        selected_sample = {
            "title": "Scientists Announce Coffee Now Cures Everything Except Monday Mornings",
            "text": "In a historic medical breakthrough, doctors confirmed that three cups of roasted coffee beans will instantly repair any broken bone, yet stubbornly fail to make morning meetings bearable."
        }
        
    if "fake_title_input" not in st.session_state:
        st.session_state.fake_title_input = ""
    if "fake_text_input" not in st.session_state:
        st.session_state.fake_text_input = ""
        
    if selected_sample:
        st.session_state.fake_title_input = selected_sample["title"]
        st.session_state.fake_text_input = selected_sample["text"]
        
    article_title = st.text_input("Article Headline / Title (optional):", value=st.session_state.fake_title_input)
    article_text = st.text_area("Article Content:", value=st.session_state.fake_text_input, height=180, placeholder="Paste the news article body here...")
    
    col_btn, _ = st.columns([1, 4])
    analyze_fake = col_btn.button("🔍 Analyze Article", type="primary", use_container_width=True)
    
    if analyze_fake and article_text.strip():
        with st.spinner("Executing ETL preprocessing and TF-IDF classification..."):
            result = predictor.predict_fake_news(article_title, article_text)
            
        if result.get("status") == "success":
            st.markdown("---")
            st.markdown("### 📋 Analysis Results")
            
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                is_fab = (result['binary_verdict'] == "Fabricated / Fake")
                badge_class = "badge-fake" if is_fab else "badge-other"
                badge_text = "🚨 BLATANT FABRICATION / FAKE" if is_fab else "ℹ️ OPINION / BIASED / CONSPIRACY"
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1.5rem;">
                    <h4>Primary Prediction Verdict</h4>
                    <div style="margin: 1rem 0;">
                        <span class="badge {badge_class}" style="font-size: 1.1rem; padding: 0.5rem 1.2rem;">{badge_text}</span>
                    </div>
                    <p style="margin-bottom: 0.3rem;"><b>Fabrication Probability:</b> {result['fabrication_probability']*100:.1f}%</p>
                    <div style="background-color: #374151; border-radius: 8px; height: 12px; width: 100%; overflow: hidden; margin-bottom: 1rem;">
                        <div style="background: linear-gradient(90deg, #F87171, #EF4444); width: {result['fabrication_probability']*100}%; height: 100%;"></div>
                    </div>
                    <p><b>Fine-Grained Category:</b> <span style="color: #60A5FA; font-weight: 700;">{result['top_category']}</span> ({result['category_confidence']*100:.1f}% match)</p>
                    <p><b>Processed Tokens:</b> {result['word_count']} words</p>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("#### 🔬 Misinformation Taxonomy Breakdown")
                breakdown = result['taxonomy_breakdown']
                df_break = pd.DataFrame(list(breakdown.items()), columns=['Category', 'Probability'])
                df_break = df_break.sort_values(by='Probability', ascending=True)
                
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='none')
                ax.barh(df_break['Category'], df_break['Probability'] * 100, color='#818CF8')
                ax.set_xlabel('Probability (%)', color='#9CA3AF')
                ax.set_facecolor('none')
                ax.tick_params(colors='#D1D5DB')
                for spine in ax.spines.values():
                    spine.set_color('#4B5563')
                st.pyplot(fig)
                
            # Trigger keywords explainability
            st.markdown("#### 🔎 Top Trigger Keywords & Contributions")
            triggers = result.get('trigger_keywords', [])
            if triggers:
                cols = st.columns(min(len(triggers), 5))
                for idx, t in enumerate(triggers[:5]):
                    cols[idx].metric(label=f"Token: '{t['word']}'", value=f"{t['weight']:+.3f}", delta=t['impact'])
            else:
                st.caption("No specific high-weighted vocabulary triggers found.")
        else:
            st.error(result.get("message", "Error analyzing article."))

# ==========================================
# TAB 2: SENTIMENT ANALYZER
# ==========================================
with tab_sent:
    st.subheader("💬 Social Media Sentiment Analysis")
    st.write("Analyze tweets, status updates, or customer feedback to detect sentiment polarity with word-level explainability.")
    
    st.markdown("**💡 Quick Test Tweets:**")
    st_col1, st_col2, st_col3 = st.columns(3)
    
    test_tweet = None
    if st_col1.button("🎉 Ecstatic / Highly Positive", use_container_width=True):
        test_tweet = "@techhub Just completed our machine learning final presentation and the jury loved it!! Best day ever 🚀🥳"
    if st_col2.button("😡 Frustrated / Highly Negative", use_container_width=True):
        test_tweet = "So disappointed with the delayed flight and lost luggage. Absolutely awful service, never booking with them again!!"
    if st_col3.button("📱 Raw Slang & Repeated Letters", use_container_width=True):
        test_tweet = "omgggg thank u sooo much for the surprise gifts, i'm sooooo happy right now <3 http://twitpic.com/abc"
        
    if "sent_text_input" not in st.session_state:
        st.session_state.sent_text_input = ""
    if test_tweet:
        st.session_state.sent_text_input = test_tweet
        
    tweet_input = st.text_area("Tweet or Social Media Post:", value=st.session_state.sent_text_input, height=120, placeholder="Type or paste a social media post here...")
    
    col_btn_sent, _ = st.columns([1, 4])
    analyze_sent = col_btn_sent.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_sent and tweet_input.strip():
        with st.spinner("Applying text normalization, mention stripping, and polarity inference..."):
            res = predictor.predict_sentiment(tweet_input)
            
        st.markdown("---")
        st.markdown("### 📋 Sentiment Breakdown")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            is_pos = (res['label'] == "Positive")
            badge_class = "badge-pos" if is_pos else "badge-neg"
            badge_text = "😊 POSITIVE SENTIMENT" if is_pos else "😡 NEGATIVE SENTIMENT"
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1.5rem;">
                <h4>Sentiment Classification</h4>
                <div style="margin: 1rem 0;">
                    <span class="badge {badge_class}" style="font-size: 1.15rem; padding: 0.5rem 1.4rem;">{badge_text}</span>
                </div>
                <p><b>Model Confidence:</b> {res['confidence']*100:.1f}%</p>
                <div style="display: flex; gap: 10px; margin-bottom: 1rem;">
                    <div style="flex: 1; text-align: center; background: #881337; padding: 0.5rem; border-radius: 8px;">
                        <small>Negative</small><br><b>{res['probabilities']['Negative']*100:.1f}%</b>
                    </div>
                    <div style="flex: 1; text-align: center; background: #065F46; padding: 0.5rem; border-radius: 8px;">
                        <small>Positive</small><br><b>{res['probabilities']['Positive']*100:.1f}%</b>
                    </div>
                </div>
                <p><b>ETL Cleaned Text:</b><br><code style="color: #FBBF24;">{res['clean_text']}</code></p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("#### 🔍 Influential Words (Explainability)")
            st.write("Weights extracted directly from Logistic Regression coefficients:")
            words = res.get('influential_words', [])
            if words:
                df_words = pd.DataFrame(words)
                df_words['Impact'] = df_words['weight'].apply(lambda w: "Positive (+)" if w > 0 else "Negative (-)")
                st.dataframe(df_words, use_container_width=True, hide_index=True)
            else:
                st.info("No specific strong polarity keywords identified.")

# ==========================================
# TAB 3: BATCH CSV PREDICTOR
# ==========================================
with tab_batch:
    st.subheader("📁 Batch File Processing")
    st.write("Upload a `.csv` file containing multiple news articles or tweets to run automated predictions in bulk.")
    
    task_mode = st.radio("Choose Task:", ["Social Media Sentiment Analysis", "Fake News Detection"], horizontal=True)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Uploaded CSV with {len(df_batch):,} rows. Preview:")
            st.dataframe(df_batch.head(4), use_container_width=True)
            
            text_column = st.selectbox("Select the column containing the text:", df_batch.columns)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch with vectorized inference..."):
                    if task_mode == "Social Media Sentiment Analysis":
                        preds = [predictor.predict_sentiment(str(t))['label'] for t in df_batch[text_column]]
                        df_batch['Predicted_Sentiment'] = preds
                    else:
                        preds = [predictor.predict_fake_news("", str(t)).get('binary_verdict', 'Unknown') for t in df_batch[text_column]]
                        df_batch['Predicted_Verdict'] = preds
                        
                st.success("Batch processing complete!")
                st.dataframe(df_batch.head(10), use_container_width=True)
                
                # Download button
                csv_data = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv_data,
                    file_name="classified_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

# ==========================================
# TAB 4: MODEL PERFORMANCE & METRICS
# ==========================================
with tab_models:
    st.subheader("📊 Rigorous Academic Model Evaluation")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("### 💬 Sentiment Analysis Model")
        sent_m = metrics.get('sentiment', {})
        st.write(f"**Architecture:** {sent_m.get('model_name', 'Logistic Regression')}")
        st.write(f"**Evaluation Split:** 16,000 unseen test tweets (stratified 50/50)")
        
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.metric("Accuracy", f"{sent_m.get('accuracy', 0)*100:.2f}%")
        stat_c2.metric("Precision", f"{sent_m.get('precision', 0)*100:.2f}%")
        stat_c3.metric("Recall", f"{sent_m.get('recall', 0)*100:.2f}%")
        stat_c4.metric("F1-Score", f"{sent_m.get('f1_score', 0)*100:.2f}%")
        
        st.markdown("#### Confusion Matrix (Sentiment)")
        cm_s = np.array(sent_m.get('confusion_matrix', [[6303, 1697], [1680, 6320]]))
        fig_s, ax_s = plt.subplots(figsize=(4.5, 3.5), facecolor='none')
        sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax_s)
        ax_s.set_ylabel('True Label', color='#9CA3AF')
        ax_s.set_xlabel('Predicted Label', color='#9CA3AF')
        ax_s.tick_params(colors='#D1D5DB')
        st.pyplot(fig_s)
        
    with m_col2:
        st.markdown("### 📰 Fake News & Misinformation Model")
        fake_bin_m = metrics.get('fake_news', {}).get('binary_detector', {})
        fake_tax_m = metrics.get('fake_news', {}).get('taxonomy_classifier', {})
        
        st.write(f"**Architecture:** {fake_bin_m.get('model_name', 'Balanced Logistic Regression')}")
        st.write(f"**Evaluation Split:** 2,403 unseen test articles")
        
        fstat_c1, fstat_c2, fstat_c3 = st.columns(3)
        fstat_c1.metric("Binary Acc.", f"{fake_bin_m.get('accuracy', 0)*100:.2f}%")
        fstat_c2.metric("Weighted F1", f"{fake_bin_m.get('f1_score', 0)*100:.2f}%")
        fstat_c3.metric("Taxonomy Acc.", f"{fake_tax_m.get('accuracy', 0)*100:.2f}%")
        
        st.markdown("#### Confusion Matrix (Fabrication Detection)")
        cm_f = np.array(fake_bin_m.get('confusion_matrix', [[120, 154], [118, 2011]]))
        fig_f, ax_f = plt.subplots(figsize=(4.5, 3.5), facecolor='none')
        sns.heatmap(cm_f, annot=True, fmt='d', cmap='Reds', xticklabels=['Other', 'Fabricated'], yticklabels=['Other', 'Fabricated'], ax=ax_f)
        ax_f.set_ylabel('True Label', color='#9CA3AF')
        ax_f.set_xlabel('Predicted Label', color='#9CA3AF')
        ax_f.tick_params(colors='#D1D5DB')
        st.pyplot(fig_f)

# ==========================================
# TAB 5: ETL ENGINEERING PIPELINE
# ==========================================
with tab_etl:
    st.subheader("⚙️ Data Engineering & ETL Pipeline Breakdown")
    st.write("Demonstration of the Extract, Transform, and Load procedures performed on the messy raw datasets.")
    
    etl_col1, etl_col2 = st.columns(2)
    with etl_col1:
        st.markdown("### 1. Social Media Sentiment (Sentiment140)")
        s_etl = etl_report.get('sentiment', {})
        st.markdown(f"""
        * **Raw File:** `training.1600000.processed.noemoticon.csv` (238 MB)
        * **Ingested Raw Rows:** `{s_etl.get('initial_rows', 1600000):,}`
        * **Duplicate Records Dropped:** `{s_etl.get('duplicates_removed', 18534):,}`
        * **Target Re-encoding:** Mapped raw target `0 -> Negative` and `4 -> Positive`.
        * **Noise Filtering:** Removed 46.1% `@mentions`, 5.0% URLs, unescaped HTML entities, and normalized elongated characters (`looooove` -> `loove`).
        * **Stratified Balance:** Saved `{s_etl.get('processed_rows', 80000):,}` balanced samples (`{s_etl.get('positive_count', 40000):,}` pos / `{s_etl.get('negative_count', 40000):,}` neg).
        * **Avg Word Length:** `{s_etl.get('avg_words_per_tweet', 12.71)}` words/tweet.
        """)
        
    with etl_col2:
        st.markdown("### 2. Fake News Corpus (`fake.csv`)")
        f_etl = etl_report.get('fake_news', {})
        st.markdown(f"""
        * **Raw File:** `fake.csv` (54 MB, 20 columns)
        * **Ingested Raw Rows:** `{f_etl.get('initial_rows', 12999):,}`
        * **Language Filter:** Removed `{f_etl.get('non_english_dropped', 596):,}` non-English articles (Russian, German, Spanish).
        * **Missing Value Imputation:** Handled 2,424 missing authors and 680 empty titles. Dropped stub bodies (&lt;30 chars).
        * **Text Fusion:** Combined `title` + `text` into single normalized representation.
        * **Deduplication:** Dropped `{f_etl.get('duplicates_removed', 148):,}` duplicate articles.
        * **Final Clean Corpus:** `{f_etl.get('clean_rows', 12011):,}` high-quality records.
        * **Avg Article Length:** `{f_etl.get('avg_words_per_article', 666.27)}` words.
        """)
        
    st.markdown("---")
    st.markdown("### 📊 Clean Misinformation Categories Distribution")
    cat_dist = f_etl.get('categories_distribution', {})
    if cat_dist:
        df_cat = pd.DataFrame(list(cat_dist.items()), columns=['Category', 'Count'])
        st.bar_chart(df_cat.set_index('Category'))

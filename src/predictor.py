import os
import joblib
import numpy as np
from src.etl import clean_tweet_text, clean_article_text

class ModelPredictor:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
            
        self.models_dir = models_dir
        
        # Load Sentiment artifacts
        self.sent_model = joblib.load(os.path.join(models_dir, "sentiment_model.joblib"))
        self.sent_vec = joblib.load(os.path.join(models_dir, "sentiment_vectorizer.joblib"))
        
        # Load Fake News artifacts
        self.fake_bin_model = joblib.load(os.path.join(models_dir, "fake_bin_model.joblib"))
        self.fake_bin_vec = joblib.load(os.path.join(models_dir, "fake_bin_vectorizer.joblib"))
        self.fake_multi_model = joblib.load(os.path.join(models_dir, "fake_multi_model.joblib"))
        self.fake_multi_vec = joblib.load(os.path.join(models_dir, "fake_multi_vectorizer.joblib"))

    def predict_sentiment(self, text: str) -> dict:
        """
        Predict sentiment for social media post / tweet.
        Includes word attribution explainability.
        """
        clean = clean_tweet_text(text)
        if not clean:
            return {
                "raw_text": text,
                "clean_text": "",
                "label": "Neutral / Empty",
                "confidence": 0.5,
                "probabilities": {"Negative": 0.5, "Positive": 0.5},
                "influential_words": []
            }
            
        vec = self.sent_vec.transform([clean])
        probs = self.sent_model.predict_proba(vec)[0]
        prob_neg, prob_pos = float(probs[0]), float(probs[1])
        
        pred_label = "Positive" if prob_pos >= 0.5 else "Negative"
        confidence = prob_pos if pred_label == "Positive" else prob_neg
        
        # Word contribution analysis
        feature_names = self.sent_vec.get_feature_names_out()
        coef = self.sent_model.coef_[0]
        
        # Find active features in this text
        nonzero_indices = vec.nonzero()[1]
        word_scores = []
        for idx in nonzero_indices:
            word = feature_names[idx]
            weight = float(coef[idx] * vec[0, idx])
            word_scores.append({
                "word": word,
                "weight": round(weight, 3),
                "orientation": "Positive" if weight > 0 else "Negative"
            })
            
        word_scores.sort(key=lambda x: abs(x['weight']), reverse=True)
        
        return {
            "raw_text": text,
            "clean_text": clean,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Negative": round(prob_neg, 4),
                "Positive": round(prob_pos, 4)
            },
            "influential_words": word_scores[:8]
        }

    def predict_fake_news(self, title: str, text: str) -> dict:
        """
        Predict whether an article is Fabricated/BS or Other Misinformation/Conspiracy,
        and classify into fine-grained misinformation taxonomy.
        """
        full_raw = f"{title}. {text}" if title else text
        clean = clean_article_text(full_raw)
        
        if not clean or len(clean.split()) < 3:
            return {
                "status": "error",
                "message": "Article text is too short to analyze accurately."
            }
            
        # 1. Binary fabrication prediction
        bin_vec = self.fake_bin_vec.transform([clean])
        bin_probs = self.fake_bin_model.predict_proba(bin_vec)[0]
        prob_other, prob_fabricated = float(bin_probs[0]), float(bin_probs[1])
        
        binary_verdict = "Fabricated / Fake" if prob_fabricated >= 0.5 else "Biased / Conspiracy / Satire"
        bin_confidence = prob_fabricated if prob_fabricated >= 0.5 else prob_other
        
        # 2. Multi-class taxonomy prediction
        multi_vec = self.fake_multi_vec.transform([clean])
        multi_probs = self.fake_multi_model.predict_proba(multi_vec)[0]
        classes = self.fake_multi_model.classes_
        
        taxonomy_breakdown = {
            cls: round(float(multi_probs[i]), 4)
            for i, cls in enumerate(classes)
        }
        top_category = max(taxonomy_breakdown, key=taxonomy_breakdown.get)
        
        # 3. Explainability: top trigger words
        feat_names = self.fake_bin_vec.get_feature_names_out()
        coef = self.fake_bin_model.coef_[0]
        nonzero_idx = bin_vec.nonzero()[1]
        
        word_scores = []
        for idx in nonzero_idx:
            word = feat_names[idx]
            weight = float(coef[idx] * bin_vec[0, idx])
            word_scores.append({
                "word": word,
                "weight": round(weight, 3),
                "impact": "Fabrication Indicator" if weight > 0 else "Contextual / Biased Marker"
            })
        word_scores.sort(key=lambda x: abs(x['weight']), reverse=True)
        
        return {
            "status": "success",
            "clean_text_preview": clean[:300] + "..." if len(clean) > 300 else clean,
            "word_count": len(clean.split()),
            "binary_verdict": binary_verdict,
            "binary_confidence": round(bin_confidence, 4),
            "fabrication_probability": round(prob_fabricated, 4),
            "top_category": top_category,
            "category_confidence": taxonomy_breakdown[top_category],
            "taxonomy_breakdown": taxonomy_breakdown,
            "trigger_keywords": word_scores[:10]
        }

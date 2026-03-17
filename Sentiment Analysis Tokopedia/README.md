# 🛒 Sentiment Analysis: Classifying Tokopedia Food & Drink Reviews

This project builds an end-to-end **NLP sentiment analysis pipeline** on real-world Indonesian e-commerce data. Customer reviews from Tokopedia's **Food & Drink** category are preprocessed, vectorized, and classified into **Positive** or **Negative** sentiment using a **Multinomial Naive Bayes** classifier — achieving **~83% accuracy** on a balanced dataset.

---

## 📦 Dataset
- **Source**: [Tokopedia Product Reviews — Food & Drink (Kaggle)](https://www.kaggle.com/datasets/kulitekno/tokopedia-product-review-category-food-and-drink)
- **Key variables**: `Review`, `Rating`
- **Language**: Indonesian (Bahasa Indonesia)
- **Task**: Binary sentiment classification — Positive (1) vs Negative (0)

---

## 📊 Visualizations & Key Insights

### 1. **Rating Distribution**
The raw dataset is heavily skewed toward **Rating 5**, which is typical for e-commerce platforms. This class imbalance needed to be addressed before modeling to prevent the classifier from defaulting to majority predictions.

![Rating Distribution](images/rating_distribution.png)

---

### 2. **Class Balancing: Before vs After Downsampling**
Sentiment labels were derived from ratings: **1–3 → Negative**, **4–5 → Positive**. The Positive class significantly outnumbered the Negative class, so the majority class was **downsampled** to match the minority count.

- Before balancing: severe Positive/Negative imbalance
- After balancing: equal class distribution for fair model training

![Class Balance](images/class_balance.png)

---

### 3. **Text Preprocessing Pipeline**
Raw reviews contain noise that degrades model performance. The following 8-step pipeline was applied before vectorization:

| Step | Operation | Reason |
|------|-----------|--------|
| 1 | Drop unused index column | Removes CSV export artifact |
| 2 | Strip whitespace characters (`\t`, `\n`, `\r`) | Normalizes spacing |
| 3 | Remove emojis & non-ASCII symbols | Focuses on textual signal |
| 4 | Lowercase conversion | Reduces vocabulary size |
| 5 | Remove punctuation & special characters | Keeps only alphanumeric tokens |
| 6 | Collapse multiple spaces | Cleans up post-strip artifacts |
| 7 | Remove duplicate rows | Prevents data leakage |
| 8 | Reset index | Ensures clean integer index |

---

### 4. **Vectorization: Bag of Words**
`CountVectorizer` was used to transform each review into a sparse vector where each dimension represents the frequency of a word in the vocabulary. CountVectorizer was chosen because it pairs naturally with Multinomial Naive Bayes, which models **word-count distributions**.

---

### 5. **Confusion Matrix — Multinomial Naive Bayes**
The confusion matrix shows a balanced performance across both classes, with no significant bias toward either sentiment label after downsampling.

![Confusion Matrix](images/confusion_matrix.png)

---

### 6. **Inference on New Reviews**
The trained model predicts sentiment on unseen reviews with a confidence score:

| Type | Review | Prediction |
|------|--------|------------|
| Positive | *produk ini bagus rasanya enak dan kualitasnya juga terbaik* | ✅ POSITIVE (confidence: ~95%) |
| Negative | *saya sangat kecewa dengan produk ini rasanya tidak enak* | ❌ NEGATIVE (confidence: ~93%) |
| Neutral-ish | *produk biasa saja tidak ada yang spesial* | ❌ NEGATIVE (confidence: ~70%) |

---

## 📈 Final Results Summary

| Metric | Negative (0) | Positive (1) |
|--------|-------------|-------------|
| Precision | 0.84 | 0.82 |
| Recall | 0.80 | 0.85 |
| F1-Score | 0.82 | 0.83 |
| **Accuracy** | | **82.79%** |

- The model achieves **~83% accuracy** on the balanced test set — strong for a simple BoW + Naive Bayes baseline.
- Both classes have comparable precision and recall, confirming the model is **not biased** toward either label after downsampling.
- The slight advantage in Positive recall (0.85 vs 0.80) is reasonable, as positive language in food reviews tends to be more distinctive.

---

## 🔭 Potential Improvements

| Idea | Expected Benefit |
|------|-----------------|
| **TF-IDF** instead of raw counts | Down-weights frequent but less informative words |
| **Stopword removal** (Indonesian) | Reduces noise from common filler words |
| **N-gram features** (bigrams/trigrams) | Captures phrases like *tidak enak* (not tasty) |
| **Stemming / lemmatization** | Reduces vocabulary by collapsing word forms |
| **IndoBERT / BERT** | Better captures contextual semantics in Indonesian |
| **SMOTE** for oversampling | Alternative to downsampling that preserves original data volume |

---

## 📚 License
This project uses publicly available data from Kaggle and is intended for educational purposes only. Sentiment labels are derived heuristically from star ratings and do not represent ground-truth human annotations.


import pandas as pd
import numpy as np
import re
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── Setup ─────────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

os.makedirs('images', exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/IMDB Dataset.csv')
print(f"Shape: {df.shape}")
print(df['sentiment'].value_counts())

# ── 2. Visualize class distribution ───────────────────────────────────────────
plt.figure(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar', color=['#4CAF50', '#F44336'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('images/class_distribution.png', dpi=150)
plt.close()
print("Saved: images/class_distribution.png")

# ── 3. Clean text ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)           # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # keep letters only
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

print("Cleaning text... (this takes 1–2 minutes)")
df['clean_review'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ── 4. Word Cloud ──────────────────────────────────────────────────────────────
try:
    from wordcloud import WordCloud
    pos_text = ' '.join(df[df['label'] == 1]['clean_review'])
    neg_text = ' '.join(df[df['label'] == 0]['clean_review'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wc_pos = WordCloud(width=700, height=400, background_color='white',
                       colormap='Greens').generate(pos_text)
    axes[0].imshow(wc_pos, interpolation='bilinear')
    axes[0].set_title('Positive Reviews – Most Frequent Words')
    axes[0].axis('off')

    wc_neg = WordCloud(width=700, height=400, background_color='white',
                       colormap='Reds').generate(neg_text)
    axes[1].imshow(wc_neg, interpolation='bilinear')
    axes[1].set_title('Negative Reviews – Most Frequent Words')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('images/wordclouds.png', dpi=150)
    plt.close()
    print("Saved: images/wordclouds.png")
except ImportError:
    print("wordcloud not installed — skipping word cloud. Run: pip install wordcloud")

# ── 5. Split data ─────────────────────────────────────────────────────────────
X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 6. TF-IDF Vectorization ───────────────────────────────────────────────────
print("Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2,
    sublinear_tf=True
)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf  = tfidf.transform(X_test)

# ── 7. Train and compare models ───────────────────────────────────────────────
print("\nTraining models...")
models = {
    'Naive Bayes':         MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    'LinearSVC':           LinearSVC(C=1.0, max_iter=2000),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train_tf, y_train)
    preds = clf.predict(X_test_tf)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"  {name:25s} → Accuracy: {acc:.4f}")

# ── 8. Best model: LinearSVC ──────────────────────────────────────────────────
best_model = models['LinearSVC']
best_preds = best_model.predict(X_test_tf)

print("\n── Classification Report (LinearSVC) ──")
print(classification_report(y_test, best_preds,
                             target_names=['Negative', 'Positive']))

# ── 9. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix – LinearSVC')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150)
plt.close()
print("Saved: images/confusion_matrix.png")

# ── 10. Model comparison bar chart ────────────────────────────────────────────
plt.figure(figsize=(8, 4))
bars = plt.bar(results.keys(), [v * 100 for v in results.values()],
               color=['#64B5F6', '#81C784', '#FF8A65'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 95)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.1,
             f"{val*100:.1f}%", ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('images/model_comparison.png', dpi=150)
plt.close()
print("Saved: images/model_comparison.png")

# ── 11. Save model ────────────────────────────────────────────────────────────
joblib.dump(best_model, 'model.pkl')
joblib.dump(tfidf,      'tfidf.pkl')
print("\nSaved: model.pkl")
print("Saved: tfidf.pkl")
print("\nAll done! Now run: streamlit run app.py")

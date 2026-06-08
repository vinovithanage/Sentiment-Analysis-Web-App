# Sentiment Analysis Web App

Predicts whether text (movie reviews, tweets, comments) is **Positive** or **Negative** using NLP and Machine Learning.


## Overview

This project builds a full NLP pipeline:
1. Text preprocessing (HTML removal, stopwords, stemming)
2. TF-IDF vectorization (unigrams + bigrams)
3. Three ML models trained and compared
4. Best model deployed as a Streamlit web app



## Model Performance

| Model               | Accuracy |
|---------------------|----------|
| Naive Bayes         | ~85%     |
| Logistic Regression | ~88%     |
| LinearSVC ✅        | ~90%     |

---

## Project Structure

```
sentiment-analysis-project/
├── app.py                     ← Streamlit web app
├── train_model.py             ← Training script (run once)
├── sentiment_analysis.ipynb   ← Full EDA + training notebook
├── model.pkl                  ← Saved LinearSVC model
├── tfidf.pkl                  ← Saved TF-IDF vectorizer
├── requirements.txt           ← Python dependencies
├── README.md
├── data/
│   └── IMDB Dataset.csv       ← Download from Kaggle
└── images/
    ├── confusion_matrix.png
    ├── model_comparison.png
    └── wordclouds.png
```


## Tech Stack

- **Python 3.9+**
- **pandas, numpy** — data handling
- **NLTK** — text preprocessing (stopwords, stemming)
- **scikit-learn** — TF-IDF, ML models, evaluation
- **Streamlit** — web app
- **matplotlib, seaborn, wordcloud** — visualizations
- **joblib** — model saving


## Author

**V.T. Thilanka Vinodani**  
B.Tech (Hons) ICT — General Sir John Kotelawala Defence University  


## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle

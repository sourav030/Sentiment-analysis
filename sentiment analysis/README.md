# 🤖 Sentiment Analysis Chatbot

A Python-based intelligent chatbot that performs real-time sentiment analysis on user conversations. This project implements both **statement-level** (Tier 2) and **conversation-level** (Tier 1) sentiment evaluation using Machine Learning.

## 📌 Project Overview
The goal of this project is to develop a conversational agent that can understand the emotional tone of a user. It uses **Natural Language Processing (NLP)** and a **Linear Support Vector Classifier (LinearSVC)** to classify text as **Positive** or **Negative**.

### ✅ Features Implemented
* **Tier 1: Conversation-Level Analysis** - Maintains full chat history and generates an overall mood report (Positive/Negative/Mixed) at the end of the session.
* **Tier 2: Statement-Level Analysis** - Instantly evaluates and displays the sentiment of every individual message sent by the user.
* **Smart Preprocessing** - Uses NLTK for stemming and stopword removal to handle noise (e.g., "loving" -> "love").
* **Context Awareness** - Utilizes **Bi-grams (N-Grams)** to understand context (e.g., correctly identifies "not good" as Negative).
* **Robust Backend** - Trained on **200,000 tweets** from the Sentiment140 dataset with ~80% accuracy.

---

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (LogesticRegression)
* **NLP Library:** NLTK (PorterStemmer, Stopwords)
* **Data Handling:** Pandas, NumPy
* **Serialization:** Pickle (For saving/loading models)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sourav030/Sentiment-analysis.git
cd SENTIMENT ANALYSIS
pip install -r requirements.txt
python src/chatbot/chatbot.py
dataset https://www.kaggle.com/datasets/kazanova/sentiment140

├── src
│   ├── chatbot
│   │   └── chatbot.py          # Main Application (Tier 1 & 2 Logic)
│   ├── ml
│   │   ├── CleanData           # Data cleaning scripts/data
│   │   ├── trained_model.sav   # Trained Brain (LinearSVC)
│   │   └── vectorizer.pkl      # Saved TF-IDF Vectorizer
├── README.md                   # Project Documentation
└── Sentiment.csv               # Source Dataset
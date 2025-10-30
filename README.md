# 📬 Spam Email Classifier

A machine learning project that classifies text messages as **Spam** or **Not Spam (Ham)** using Python and Scikit-learn.

---

## 🚀 Features
- Preprocesses text using CountVectorizer
- Trains a Multinomial Naive Bayes model
- Evaluates model accuracy and metrics
- Saves the model for later use (`spam_model.pkl`)
- Allows real-time user message classification

---

## 🧠 Dataset
Uses the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).  
Download it and place as `spam.csv` in the project folder.

---

## ⚙️ Installation
```bash
pip install -r requirements.txt

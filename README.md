📰 News Classification using NLP

📘 Project Overview

This project focuses on automated news classification using Natural Language Processing (NLP) and Machine Learning. It analyzes text data to categorize news articles into predefined topics, leveraging preprocessing, feature extraction, and model training to achieve high accuracy.

⚙️ Features

  1. Preprocesses raw text (cleaning, tokenization, TF-IDF vectorization).

  2. Classifies news into 4 categories: World, Sports, Business, Science.

  3. Simple Streamlit web app for interactive classification.

  4. Achieves 88% accuracy using SVM.

🧠 Objective

To build a machine learning model that accurately classifies news articles based on textual content using TF-IDF and multiple NLP algorithms.

🔄 Workflow

  1. Data Preprocessing: Tokenization, stopword removal, lemmatization.

  2. Feature Extraction: TF-IDF vectorization converts text into numerical features.

  3. Model Training: Implemented Logistic Regression, SVM, Naive Bayes, Decision Tree, and SGD Classifier.

  4.Evaluation: Compared models and selected the best performing one (SVM).

📊 Model Performance

| Model               | Accuracy (%) |
|----------------------|--------------|
| Logistic Regression  | 87.59%       |
| SGD Classifier       | 87.86%       |
| Decision Tree        | 72.88%       |
| Naive Bayes          | 85.34%       |
| SVM Classifier       | **87.92%**   |


🧩 Tech Stack

  1. Language: Python

  2. Libraries: NumPy, Pandas, Scikit-learn, NLTK, Matplotlib, Streamlit

  3. Techniques: TF-IDF Vectorization, Lemmatization, Stopword Removal

  4. Algorithms: SVM, Logistic Regression, Naive Bayes, Decision Tree


📈 Results

The SVM Classifier achieved 88% accuracy in categorizing news into 4 major categories.
This project demonstrates the effectiveness of classical NLP and ML pipelines for real-world text classification tasks.

🚀 Future Work

Integrate deep learning models (LSTM / BERT).

Expand dataset for multilingual news sources.

Deploy Streamlit app on cloud (Streamlit Cloud / Hugging Face).

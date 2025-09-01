📰 News Article Classifier

The News Article Classifier is a machine learning project that automatically classifies news articles into categories such as World, Sports, Business, and Science.
It uses Natural Language Processing (NLP) techniques and a trained Logistic Regression model (88% accuracy) to analyze and categorize text.

📂 Project Structure
News-Article-Classifier/
│── app.py               # Streamlit app for user interface
│── svm_nlp.pkl        # Trained ML model
│── vectorizer_svm.pkl       # TF-IDF vectorizer
│── requirements.txt     # Required dependencies
│── README.md            # Project documentation


⚙️ Features :

Preprocesses raw text (cleaning, tokenization, TF-IDF vectorization).

Classifies news into 4 categories: World, Sports, Business, Science.

Simple Streamlit web app for user interaction.

Achieves 88% accuracy using Logistic Regression.

📊 Model Training :

Used TF-IDF vectorization for feature extraction.
Trained multiple models: Logistic Regression, SGD Classifier, Decision Tree, Naive Bayes, and SVM.
Logistic Regression achieved the best accuracy (88.12%), and was chosen for deployment.

🚀 Future Enhancements :

Add more categories (e.g., Technology, Entertainment, Politics).
Train deep learning models (LSTMs / Transformers).
Deploy on cloud platforms (Heroku, AWS, GCP).


Requirements :

Python 3.9+
Streamlit
Scikit-learn
NLTK
Pandas, NumPy



Author
   Developed by Sai Teja ✨

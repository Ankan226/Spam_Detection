# Spam_Detection

📧 Spam Classifier Web App

A machine learning-powered web application built with Flask that classifies text messages as Spam or Not Spam in real-time.
This project demonstrates end-to-end NLP pipeline integration — from text preprocessing to model deployment using Flask.

**Features**

🧠 Trained ML Model (using scikit-learn)

🔤 TF-IDF Vectorization for text feature extraction

🧹 Text Preprocessing: tokenization, stopword removal, stemming

🌐 Interactive Web Interface built with Flask

⚡ Fast, lightweight, and easy to deploy

**Project Structure**
Spam-Classifier/
│
├── app.py                   # Flask backend application
├── model.pkl                # Trained ML model
├── vectorizer.pkl           # TF-IDF vectorizer
├── spam.csv                 # Dataset used for training
├── spam_classifier.ipynb    # Jupyter notebook (EDA + model training)
├── requirements.txt         # Python dependencies
└── templates/
    └── index.html           # Frontend HTML page

🔧 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate        # For Linux/Mac
venv\Scripts\activate           # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Download NLTK Data

Open a Python shell and run:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

5️⃣ Run the Flask App
python app.py


Then open your browser and go to:

http://127.0.0.1:5000/

**How It Works**

Input Text → User enters a message in the input box.

Text Transformation →

Converts text to lowercase

Tokenizes using NLTK

Removes punctuation and stopwords

Applies stemming

Vectorization → Converts processed text into numerical form using TF-IDF.

Prediction → Model predicts Spam (1) or Not Spam (0).

Output → Result is displayed on the webpage.

**Model Details**

Algorithm: Multinomial Naive Bayes

Vectorizer: TF-IDF (Term Frequency–Inverse Document Frequency)

Training Dataset: spam.csv (SMS Spam Collection Dataset)

Accuracy: ~97% (depending on preprocessing and split)

**Requirements**

All dependencies are listed in requirements.txt:

nltk
scikit-learn
Flask
Flask-Cors
pandas
numpy


Install them via:

pip install -r requirements.txt

**UI Preview (Example)**
--------------------------------------------
|   📧  Enter your message below:           |
|   [ Hey, you won $1000! Click here ]     |
|                                           |
|   [ Predict ]                            |
--------------------------------------------
Result: 🚨 Spam

🛠️ Future Enhancements

Add email spam classification support

Integrate REST API endpoints for external access

Use deep learning models (LSTM/BERT) for better accuracy

Deploy on Render / Heroku / AWS / Railway

**References**

NLTK Documentation

Scikit-learn Documentation

Flask Documentation

**Author**

Ankan Pal
📍 IIT Patna | Backend Developer | AI & ML Enthusiast

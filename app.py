import os
import pickle
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# ====================================
# Initialize Flask app
# ====================================
app = Flask(__name__)

# ====================================
# NLTK setup (Render compatible)
# ====================================
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Helper for safe downloads
def safe_nltk_download(package):
    try:
        nltk.data.find(package)
        print(f"NLTK resource '{package}' already exists.")
    except LookupError:
        print(f"Downloading missing NLTK resource: {package}")
        nltk.download(package.split("/")[-1], download_dir=nltk_data_path)

# Download essential tokenizers/corpora
safe_nltk_download("corpora/stopwords")
safe_nltk_download("tokenizers/punkt")
safe_nltk_download("tokenizers/punkt_tab")

ps = PorterStemmer()

# ====================================
# Load model and vectorizer (safe load)
# ====================================
def safe_load_pickle(filename):
    try:
        print(f"Loading {filename} from:", os.getcwd())
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

tfidf = safe_load_pickle("vectorizer.pkl")
model = safe_load_pickle("model.pkl")

if tfidf is None or model is None:
    print("Critical: vectorizer.pkl or model.pkl missing or corrupt.")
else:
    print("Model and Vectorizer loaded successfully.")

# ====================================
# Text preprocessing
# ====================================
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ====================================
# Prediction function
# ====================================
def predict_spam(input_sms):
    if tfidf is None or model is None:
        return "Error: Model not loaded properly."

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return "Spam" if result == 1 else "Not Spam"

# ====================================
# Routes
# ====================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_sms = request.form["message"]
        result = predict_spam(input_sms)
        return render_template("index.html", result=result)

# ====================================
# Main entry point
# ====================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
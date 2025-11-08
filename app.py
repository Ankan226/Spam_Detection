# Main entry point for our project
import pickle
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

#Flask app - starting point of our api
app = Flask(__name__)

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

def predict_spam(input_sms):
    # Transform the input text
    transformed_sms = transform_text(input_sms)
    print(f"Transformed text: {transformed_sms}")
    
    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    print(f"Vector shape: {vector_input.shape}")
    
    # Make prediction
    result = model.predict(vector_input)[0]
    probability = model.predict_proba(vector_input)[0]
    
    print(f"Prediction result: {result}")
    print(f"Prediction probabilities: {probability}")
    
    if result == 1:
        return "Spam"
    else:
        return "Not Spam"


@app.route('/') #homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # predict route
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        print(f"Final result being passed to template: '{result}'")
        print(f"Result type: {type(result)}")
        return render_template('index.html', result = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


# localhost ip address = 0.0.0.0:5000
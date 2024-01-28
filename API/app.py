from flask import Flask, jsonify, request
from flask_restx import Resource, Api
import collections
import joblib
import pandas as pd

app = Flask(__name__)
api = Api(app)

model = joblib.load('phishing_classifier.joblib')
data = pd.read_csv('emails.csv').drop('Email No.', axis=1)


def preprocess(email):
    def count_words(email):
        # Tokenize the email into words using a regular expression
        words = email.split()
        # Count the frequency of each word
        word_count = collections.Counter(words)
        return word_count

    # Count the words in the email
    email_word_count = count_words(email)

    data.columns = data.columns.astype(str)

    # Create a new dataframe row with the same columns, initialized to 0
    new_email = pd.DataFrame([0]*(len(data.columns) - 1), index=data.drop(['Prediction'], axis=1).columns).transpose()

    # Update the counts in the new row using the email word count
    for word, count in email_word_count.items():
        if word in new_email.columns:
            new_email.loc[0, word] = count
    

@app.route('/predict', methods=['GET'])
def predict():
    # Get the email string from the query parameter
    email_body = request.args.get('email')  

    # Preprocess email_body to match the model input format
    feature_vector = preprocess(email_body)
    
    # Make a prediction
    prediction = model.predict([feature_vector])
    
    # Return the result as JSON
    return jsonify({'is_phishing': bool(prediction[0])})
    

if __name__ == '__main__':
    app.run(debug=True)
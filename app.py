import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

df = pd.read_csv("truefake.csv", low_memory=False)
df = df.dropna()

X = df['headline'] 
y = df['label']

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=1.0, min_df=0.01)
X_tfidf = tfidf_vectorizer.fit_transform(X)

log_reg_model = LogisticRegression()
log_reg_model.fit(X_tfidf, y)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None) 

@app.route('/predict', methods=['POST'])
def predict():
    headline = request.form['headline']
    headline_tfidf = tfidf_vectorizer.transform([headline])
    prediction = log_reg_model.predict(headline_tfidf)

    result = "Fake" if prediction[0] == 0 else "Real"
    
    return render_template('index.html', prediction=result, headline=headline)

if __name__ == '__main__':
    app.run(debug=True)

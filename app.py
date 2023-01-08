from flask import Flask, render_template, request
from textblob import TextBlob
import pickle
from gensim.summarization import keywords

# create Flask app
app = Flask(__name__)

# load pickle models
model = pickle.load(open('svc.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        # get review from form
        review = request.form['review']

        # create TextBlob object from review
        blob = TextBlob(review)

        # get sentiment and score
        sentiment = blob.sentiment.polarity
        score = round(sentiment, 2)

        # vectorize review
        review_vec = vectorizer.transform([review])

        # predict sentiment
        sentiment = model.predict(review_vec)[0]

        # determine sentiment emojis
        if sentiment == 1:
            emojis = "Positive 😊"
        else:
            emojis = "Negative 😞"

        words = keywords(review,words = 10)
        cleaned_string = words.replace("\\n", " ").replace("\\", " ")
        keyword = cleaned_string.strip("\"")
        # print(keyword)
        # print(type(keyword))

        return render_template('sentiment_analysis.html', review=review, emojis=emojis, score=score,keyword = keyword)
    return render_template('sentiment_analysis.html')

if __name__ == '__main__':
    app.run(debug = True)

import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import string
from flask_ngrok import run_with_ngrok
from plotly.graph_objs import Pie
import pickle
app = Flask(__name__)
run_with_ngrok(app)

# load data
df = df = pd.read_csv('dataset.csv')
df=df.drop('Unnamed: 0',axis=1)


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def tokenize(text):
    """
    Split the input sentence into tokens. Following steps are taken here:
        remove punctuations -> lowercase -> remove stop words -> lemmatize
    """
    translator = str.maketrans('', '', string.punctuation)
    # normalize
    text = text.strip().lower().translate(translator)
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens if token not in stop_words]
    return tokens


# load model
file1 = open("stacking.model",'rb')
model = pickle.load(file1)


@app.route('/')
@app.route('/index')
def index():
    graphs = [
        {
            'data': [
                Pie(values=df['label'].value_counts(), labels=['Normal', 'Sarcastic'])
            ],

            'layout': {
                'title': 'Distribution of sarcasm Types'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_label = str(model.predict([query])[0])
    print(classification_label, type(classification_label))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_label
    )


def main():
    app.run()


if __name__ == '__main__':
    main()

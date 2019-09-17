import pickle
from flask import Flask, render_template, request, redirect, url_for
from flask.json import jsonify
from flask_wtf import Form
from wtforms import StringField, PasswordField, FloatField
from wtforms.validators import InputRequired
import math
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'DontTellAnyone'

special_phrases = {'homework': 'hw',
                   'problem set': 'pset',
                   'long test': 'lt',
                   'long exam': 'lt'}
teachers = ['marcelo', 'rmarcelo', 'bataller', 'zosa', 'eden', 'reyes',
            'antonio', 'mina', 'marasigan', 'go', 'nable', 'loyola', 'lee-chua',
            'vistro-yu', 'sarmiento', 'cuajunco', 'david', 'delaspenas',
            'tabares', 'bautista', 'dbautista', 'deleoz', 'guillermo', 'justan',
            'muga', 'pangilinan', 'quimpo', 'santos', 'teng', 'angeles',
            'torres', 'torres', 'martin', 'chanshio', 'delaratuprio', 'verzosa',
            'nebres', 'provido', 'garces', 'cabral', 'tolentino', 'garciano',
            'ruiz', 'briones', 'francisco', 'miro', 'yap', 'aberin', 'dayao',
            'guzon', 'delavega', 'mallari', 'manibo', 'domingo']

class SearchQueryForm(Form):
    search_query = StringField("Query", validators=[InputRequired()])

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchQueryForm()
    return render_template('index.html', form=form, res=[])

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchQueryForm()
    if form.validate_on_submit():
        res = search_query(form.search_query.data)
        return render_template('index.html', form=form, res=res)
    return redirect(url_for('index'))

def to_link(filename):
    return 'https://storage.cloud.google.com/ams-lts/' + filename[41:].replace('\\', '/').replace(' ', '%20')

def tokenize(query):
    # also separate by /
    query = query.lower()

    tokens = []
    for special_phrase in special_phrases:
        used_suffix = False
        for i in range(1, 7):
            special_phrase_i = special_phrase + ' ' + str(i)
            if special_phrase_i in query:
                tokens.append(special_phrases[special_phrase] + str(i))
                query = query.replace(special_phrase_i, ' ')
                used_suffix = True
                break
        if not used_suffix and special_phrase in query:
            tokens.append(special_phrases[special_phrase])
            query = query.replace(special_phrase, ' ')
            break

    tokens += query.replace(',', ' ').replace('\\', ' ').split()
    return tokens

def count(query, corpus_tokens):
    return corpus_tokens.count(query)

def tf_score(query, corpus_tokens):
    return corpus_tokens.count(query) / len(corpus_tokens)

def tf_binscore(query, corpus_tokens):
    if query in corpus_tokens:
        if query in teachers:
            return 2
        return 1
    return 0
    #return (1 if query in corpus_tokens else 0) + \
    #       (1 if query in teachers else 0)

def search_query(query):
    with open('database.pickle', 'rb') as pickle_off:
        data = pickle.load(pickle_off)

    query_tokens = tokenize(query)
    print(query_tokens)

    idf_scores = {}
    for token in query_tokens:
        cnt = 0
        for file in data:
            filename_tokens = tokenize(file['filename'])
            content_tokens = tokenize(file['content'])
            if token in filename_tokens or token in content_tokens:
                cnt += 1
        idf_scores[token] = math.log(len(data)/(cnt+1))

    hit_counts = []
    for file in data:
        bin_hits = []
        cnt_hits = []
        filename_tokens = tokenize(file['filename'])
        content_tokens = tokenize(file['content'])

        for token in query_tokens:
            bin_hits.append(tf_binscore(token, filename_tokens + content_tokens))
            cnt_hits.append(count(token, filename_tokens + content_tokens))

        a = np.asarray(list(tf_binscore(token, query_tokens) for token in query_tokens))
        b = np.asarray(bin_hits)
        cosine_similarity = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

        hit_counts.append((cosine_similarity,
                           bin_hits,
                           cnt_hits,
                           to_link(file['filename']),
                           file['filename'][41:],
                           file['content'][:min(150, len(file['content']))] + '...'
                           ))
    hit_counts.sort(key=lambda tup: (tup[0], tup[1], tup[2]), reverse=True)
    return hit_counts[:20]

if __name__ == '__main__':
    app.run()

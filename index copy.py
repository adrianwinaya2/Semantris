from flask import Flask, render_template, request, session
import json
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random as rand
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import ast

app = Flask(__name__)
app.secret_key = 'semantris_secret_key'

# Load the model
model = hub.load('model_text')
corpus = pd.read_csv('common_words.csv')

# Game Settings
pop_boundary = 4
max_words = 10

# ! FUNCTIONS
def similarity(target, word):
    vectors = model([target, word])
    vector1 = vectors[0].reshape(1, -1)
    vector2 = vectors[1].reshape(1, -1)

    return cosine_similarity(vector1, vector2)[0][0]

def fill_words(words, history, target):
    while len(words) < max_words:
        index = rand.randint(1, corpus.shape[0] - 1)
        word = corpus['common_noun'][index]
        if word.lower() not in history:
            words.insert(0, word)
            history.append(word)
    target = rand.choice(words[:-4])
    return words, history, target

# ! VARIABLE SETTINGS):
@app.before_request
def remove_error():
    session['error'] = ''
    
# ! ROUTING
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/play2', methods=['POST'])
def play2():
    session['score'] = 0
    session['target'] = ''
    session['words'] = []
    session['history'] = []
    session['error'] = ''

    session['words'], session['history'], session['target'] = fill_words(session['words'], session['history'], session['target'])
    return render_template('play_async3.html')

@app.route('/check', methods=['POST'])
def play():

    answer = str(request.form['answer']).lower()
    if answer in session['words']:
        session['error'] = {'message': 'Word already exists!', 'err_word': answer}
        return render_template('play_async3.html')
        
    words_dict = {w: similarity(answer, w) for w in session['words']}
    sorted_dict = dict(sorted(words_dict.items(), key=lambda x: x[1]))
    sorted_arr = list(sorted_dict.keys())

    if session['target'] not in sorted_arr:
        sorted_dict.pop()  # Remove the last word
        sorted_dict[session['target']] = None  # Add target to the list

    # If the target is top 4
    target_index = sorted_arr.index(session['target'])
    rank = len(sorted_arr) - target_index

    if rank <= pop_boundary:
        sorted_arr = sorted_arr[target_index + 1:]
        session['score'] += target_index + 1
        session['words'], session['history'], session['target'] = fill_words(sorted_arr, session['history'], session['target'])
        print(session['words'])

    return render_template('play_async3.html')

if __name__ == '__main__':
    # Mac OS kadang nabrak port 5000 maka pakai port 8000
    app.run(host="0.0.0.0", port=8000, debug=True)

    # Untuk Windows
    # app.run()

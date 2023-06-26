from flask import Flask, render_template, request, jsonify, session, send_file
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

# Load the common words data
corpus = pd.read_csv('common_words.csv')

# Game Settings
pop_boundary = 4
max_words = 10

def embed(input):
    return model(input)

def similarity(target, word):
    vectors = model([target, word])
    vector1 = vectors[0]
    vector2 = vectors[1]
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]

def fill_words(words, history, target):
    while len(words) < max_words:
        index = rand.randint(1, corpus.shape[0] - 1)
        word = corpus['common_noun'][index]
        if word.lower() not in [w.lower() for w in history]:
            words.insert(0, word)
            history.append(word)
    target = rand.choice(words[:-pop_boundary])
    return words, history, target

@app.before_request
def remove_error():
    session['error'] = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gameover')
def gameOver():
    return render_template('gameover_jup.html')

@app.route('/play', methods=['POST'])
def play():
    session['score'] = 0
    session['target'] = ''
    session['words'] = []
    session['history'] = []
    session['error'] = ""

    session['words'], session['history'], session['target'] = fill_words(session['words'], session['history'], session['target'])

    return render_template('play_jup.html')


@app.route('/check', methods=['POST'])
def check():
    # Get the answer from the form data
    answer = str(request.form['answer']).lower()
    
    if answer in (word.lower() for word in session['words']):
        return jsonify({
            'score': session['score'],
            'words': session['words'],
            'target': session['target'],
            'error': 'Word already exists!'
        })
    
    else:
        print("Unsorted : ", session['words'])
        words_dict = {w: similarity(answer, w) for w in session['words']}
        sorted_dict = dict(sorted(words_dict.items(), key=lambda x: x[1]))
        sorted_arr = list(sorted_dict.keys())
        print(f"Sorted : ${sorted_arr}")

        # If the target is top 4
        target_index = sorted_arr.index(session['target'])
        rank = len(sorted_arr) - target_index
        print(rank)

        if rank <= pop_boundary:
            del sorted_arr[-pop_boundary:target_index+1]
            session['score'] += (5 - rank)
            session['words'], session['history'], session['target'] = fill_words(sorted_arr, session['history'], session['target'])
        else:
            session['words'] = sorted_arr
        
        return jsonify({
            'score': session['score'],
            'words': session['words'],
            'target': session['target']
        })

# Route to serve the video file
@app.route('/background.mp4')
def send_video():
    return send_file('templates/background.mp4', mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

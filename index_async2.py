from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
import json
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random as rand
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import requests

app = Flask(__name__)
app.secret_key = 'semantris_secret_key'

cors = CORS(app, resources={r"/main/api_score": {"origins": "*"}})

# Load the model
model = hub.load('model_text')

# Load the common words data
corpus = pd.read_csv('common_words.csv')

# Game Settings
pop_boundary = 4 # Untuk menghapus 4 kata terbawah
max_words = 10 # Untuk menentukan jumlah kata yang akan ditampilkan

team_name = ''

def embed(input):
    return model(input)

# Untuk mencari similarity antara 2 kata
def similarity(target, word):
    # Mengubah kata menjadi vektor
    vectors = model([target, word])
    vector1 = vectors[0].reshape(1, -1)
    vector2 = vectors[1].reshape(1, -1)

    return cosine_similarity(vector1, vector2)[0][0]

# Mengisi kata - kata untuk ditampilkan
def fill_words(words, history, target):
    while len(words) < max_words:
        index = rand.randint(1, corpus.shape[0] - 1)
        word = corpus['common_noun'][index].lower()

        # Mengecek apakah kata sudah pernah ditampilkan
        if word not in [w.lower() for w in history]:
            words.insert(0, word)
            history.append(word)
    target = rand.choice(words[:-pop_boundary])
    return words, history, target

# ! OTHERS
@app.before_request
def remove_error():
    session['error'] = ''

# Web di run mulai dari index
@app.route('/')
def index():
    return render_template('index.html')

# Routing ke web game over
@app.route('/gameover')
def gameOver():
    return render_template('gameover_jup.html')

# Route untuk mengambil file mp4
@app.route('/background.mp4')
def send_video():
    return send_file('templates/background.mp4', mimetype='video/mp4')

@app.route('/staticbackground.png')
def send_image():
    return send_file('templates/staticbackground.png', mimetype='image/png')

@app.route('/proxy/api_score', methods=['POST'])
def proxy_api_score():
    # Get the request data from the frontend
    request_data = request.get_json()
    
    # Define the target URL (the remote API)
    target_url = 'https://irgl.petra.ac.id/main/api_score'
    
    # Forward the POST request to the remote server
    response = requests.post(target_url, json=request_data)
    
    # Return the response from the remote server to the frontend
    return jsonify(response.json()), response.status_code

# ! ROUTING
@app.route('/play', methods=['POST'])
def play():
    # Initialize session
    session['team_name'] = ''
    session['score'] = 0
    session['target'] = ''
    session['words'] = []
    session['history'] = []
    session['error'] = ''

    # Mengisi words buat di output
    session['words'], session['history'], session['target'] = fill_words(session['words'], session['history'], session['target'])

    return render_template('play_async2.html')


# Routing buat ngecheck
@app.route('/check', methods=['POST'])
def check():
    # Get the answer from the form data
    answer = str(request.form['answer']).lower()
    
    # Check if the answer is correct
    if answer not in [word.lower() for word in session['words']]:
        # Menghitung similarity nya
        words_dict = {w: similarity(answer, w) for w in session['words']}
        
        # Sorting dictionary sesuai dengan similarity dengan answer
        sorted_dict = dict(sorted(words_dict.items(), key=lambda x: x[1]))
        sorted_arr = list(sorted_dict.keys())
        sorted_list = sorted_arr.copy()

        print("Unsorted : ", session['words'])
        print(f"Sorted : {sorted_arr}")

        # If the target is top 4
        target_index = sorted_arr.index(session['target'])
        rank = len(sorted_arr) - target_index

        # Menghapus isi list yang sudah benar dan menambah score serta mengisi kata baru
        if rank <= pop_boundary:
            del sorted_arr[-pop_boundary:target_index+1]
            session['score'] += (5 - rank)
            session['words'], session['history'], session['target'] = fill_words(sorted_arr, session['history'], session['target'])
            status = 'success'
        else:
            session['words'] = sorted_list
            status = 'fail'
        
        return jsonify({
            'score': session['score'],
            'words': session['words'],
            'sorted': sorted_list,
            'target': session['target'],
            'error': '',
            'status': status
        })
    
    return jsonify({
        'score': session['score'],
        'words': session['words'],
        'sorted': [],
        'target': session['target'],
        'error': 'Word already exists!',
        'status': ''
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

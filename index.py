from flask import Flask, render_template, request
import json
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random as rand
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

app = Flask(__name__)

# Load the model
model = hub.load('model_text')

# Load the common words data
data_common = pd.read_csv('common_words.csv')

def embed(input):
    return model(input)

def similarity(target, word):
    vectors = model([target, word])
    vector1 = vectors[0]
    vector2 = vectors[1]
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]

words_arr = []

# Menjaga isi tetap 10
while len(words_arr) < 10:
    index = rand.randint(1, data_common.shape[0] - 1)
    word = data_common['common_noun'][index]
    if word.lower() not in [w.lower() for w in words_arr]:
        words_arr.append(word)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    score = 0
    out_array = {}

    # Initialize out_array with words_arr
    for i in words_arr:
        out_array[i] = None

    # Random target
    target = rand.choice(list(out_array.keys()))

    return render_template('play.html', score=score, words=list(out_array.keys()), target=target)


@app.route('/check', methods=['POST'])
def check():
    answer = request.form['answer']
    score = int(request.form['score'])
    out_array = {}

    # Get the current state from the form data
    out_array = request.form['out_array']

    # Check answer
    if answer.lower() not in (word.lower() for word in out_array):
        # Check similarity per word with answer
        for i in out_array:
            out_array[i] = similarity(answer, i)

        # Sort dict
        out_array = dict(sorted(out_array.items(), key=lambda x: x[1]))

        # Check target placement
        listed_out_array = list(out_array.keys())
        target = request.form['target']
        if target not in listed_out_array:
            out_array.pop(listed_out_array[-1])  # Remove the last word
            out_array[target] = None  # Add target to the list

        # Check if the target is in the list again
        target_index = listed_out_array.index(target)
        if target_index > len(out_array) - 5:
            # Remove target
            for i in range(target_index, len(out_array) - 5, -1):
                del out_array[listed_out_array[i]]
                score += 1

            # Ensure there are still 10 words in the array
            while len(out_array) < 10:
                index = rand.randint(1, data_common.shape[0] - 1)
                word = data_common['common_noun'][index]
                if word not in out_array:
                    out_array[word] = None

            # Random target
            target = rand.choice(list(out_array.keys()))

        # Convert out_array to JSON string
        out_array_json = json.dumps(listed_out_array)

        return render_template('play.html', score=score, words=out_array.keys(), target=target, out_array=out_array_json)

    else:
        return render_template('play.html', score=score, words=out_array.keys(), target=request.form['target'], error='Word already in array')


if __name__ == '__main__':
    # Mac OS kadang nabrak port 5000 maka pakai port 8000
    app.run(host="0.0.0.0", port=8000, debug=True)

    # Untuk Windows
    # app.run()

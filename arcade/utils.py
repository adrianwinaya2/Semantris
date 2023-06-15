from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def get_sorted_dict():
    model = hub.load('model_text')

    def embed(input):
        return model(input)

    def similarity(target, word):
        vectors = model([target, word])

        vector1 = vectors[0]
        vector2 = vectors[1]

        vector1 = vector1.reshape(1, -1)
        vector2 = vector2.reshape(1, -1)

        return cosine_similarity(vector1, vector2)[0][0]

    target = 'show'
    arr = ['rain', 'library', 'toothbrush', 'sun', 'television', 'umbrella', 'website', 'refrigerator', 'shark', 'bath']
    out_array = {}

    for i in arr:
        out_array[i] = similarity(target, i)

    sorted_dict = dict(sorted(out_array.items(), key=lambda x: x[1]))
    return sorted_dict

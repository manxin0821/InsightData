#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import tensorflow as tf
#from tensorflow import keras
# from keras import regularizers, Sequential
# from keras.layers import Embedding, GlobalAveragePooling1D, Dense
# from keras.callbacks import Callback, EarlyStopping
# from keras.optimizers import SGD
from keras.models import load_model, model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
# import numpy as np
# from numpy import ndarray
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score
# from sklearn.utils import class_weight
# from sklearn.utils.validation import check_is_fitted, column_or_1d
# from cophi_toolbox import preprocessing
# import pandas as pd
# from hatesonar import Sonar

# from collections import defaultdict
# from operator import itemgetter
# import bisect
# import matplotlib.pyplot as plt
# import pickle
import importlib
import util
importlib.reload(util)
from util import make_w2v_embeddings, split_and_zero_padding, ManDist

#
# from collections import defaultdict
# import bisect
import pickle
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from utils_webapp import pred

from keras import backend as K

#from deployment import graph
#graph.append(tf.get_default_graph())

# model


# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Load models.....

modelpath='./en_SiameseLSTM.h5'
datapath='.'


json_file = open('sentiment_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    loaded_model_sentiment = model_from_json(loaded_model_json)
    loaded_model_sentiment.load_weights('sentiment_model.h5')
with open('encoder.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
encoder=unserialized_data['encoder']


global model
model = load_model(modelpath, custom_objects={"ManDist": ManDist})
model._make_predict_function()
global graph
graph = tf.get_default_graph()
# end of model loading.....

@app.route('/')
def homepage():
    """Displays the homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def route_predict():

    # senti should be 0 or 1, 0 is negative, 1 is positive

    string_input=request.get_json(force=True)
    input_sentence = string_input['text']
    print(111)
    #results = os.exec('python test.py input_sentence')

    with graph.as_default():
        output = []
        results=pred(input_sentence,model,datapath)
        output.append(results[0][1])
        output.append(results[1][1])
        output.append(results[2][1])
        
        print(results)
        
        #return jsonify('a')
        return jsonify(output)
    
    K.clear_session()




if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0',port=80)
    app.run(debug=True, host='0.0.0.0', port=9000)





from __future__ import absolute_import, division, print_function
import tensorflow as tf
#from tensorflow import keras
from keras import backend as K
from keras import regularizers, Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import SGD
from keras.models import load_model, model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn.utils.validation import check_is_fitted, column_or_1d
from cophi_toolbox import preprocessing
import pandas as pd
from hatesonar import Sonar
sonar = Sonar()
from collections import defaultdict
from operator import itemgetter
import bisect
import matplotlib.pyplot as plt
import pickle
import importlib
import util
importlib.reload(util)
from util import make_w2v_embeddings, split_and_zero_padding, ManDist

json_file = open('sentiment_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    loaded_model_sentiment = model_from_json(loaded_model_json)
    loaded_model_sentiment.load_weights('sentiment_model.h5')
with open('encoder.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
encoder=unserialized_data['encoder']


# senti should be 0 or 1, 0 is negative, 1 is positive

def compareSim(input_sentence, model, datapath, senti):
	# retrive reviews based on input sentiment
	if senti == 0:
		datafile = datapath + '/yelp_0.txt'
	elif senti == 1:
		datafile = datapath + '/yelp_1.txt'

	with open(datafile) as f:
		datayelp = f.readlines()

	# data cleaning
	for i in range(len(datayelp)):
		datayelp[i] = datayelp[i].replace('.', '').replace('\n', '').replace('!', '')

	# find the top 3 similar reviews
	result_index = []
	test_sentence_pairs = []
	for i in range(len(datayelp)):
		test_sentence = (input_sentence, datayelp[i])
		test_sentence_pairs.append(test_sentence)

	embedding_dict = {}

	test_df = pd.DataFrame(test_sentence_pairs, columns=['question1', 'question2'])
	for q in ['question1', 'question2']:
		test_df[q + '_n'] = test_df[q]

	test_df, embeddings = make_w2v_embeddings(embedding_dict, test_df, embedding_dim=300)

	X_test = split_and_zero_padding(test_df, 10)

	assert X_test['left'].shape == X_test['right'].shape

	preds = list(model.predict([X_test['left'], X_test['right']]))

	results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
	results.sort(key=itemgetter(2), reverse=True)

	return results[0:3]


def check_senti(input_sentence):
	words = list(preprocessing.tokenize(input_sentence))
	words = list(map(lambda s: 'unknown' if s not in encoder.classes_ else s, words))
	encoder_classes = encoder.classes_.tolist()
	bisect.insort_left(encoder_classes, 'unknown')
	encoder.classes_ = np.array(encoder_classes)
	word_idx = np.array([encoder.transform(words)])

	if loaded_model_sentiment.predict_proba(word_idx) > 0.6:
		senti = 1
	else:
		senti = 0

	return senti


def pred(input_sentence, model, datapath):
	class_value = sonar.ping(text=input_sentence)['top_class']
	output=[]
	if class_value == 'offensive_language':
		senti = check_senti(input_sentence)
		results = compareSim(input_sentence, model, datapath, senti)
		output.append(results[0][1]+'#')
		output.append(results[1][1]+'#')
		output.append(results[2][1]+'#')
	else:
		output = ['NO TOXIC']
	return output

modelpath='./en_SiameseLSTM.h5'
datapath='.'
model = load_model(modelpath, custom_objects={"ManDist": ManDist})
#input_sen='The food is damn good'
#print(pred(input_sen,model,datapath))



if __name__ == '__main__':
    import sys
    #print(sys.argv[1])
    print(pred(sys.argv[1], model, datapath))



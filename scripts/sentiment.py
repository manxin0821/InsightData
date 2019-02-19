from __future__ import absolute_import, division, print_function
from keras import Sequential
import keras.backend as K
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.optimizers import Adam
from keras.models import model_from_json, load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import pandas as pd
import numpy as np
from util import make_w2v_embeddings, split_and_zero_padding, f1_score, auc
from hatesonar import Sonar
import preprocessing
from collections import defaultdict
import bisect
import re
import logging
import matplotlib.pyplot
import pickle
from pathlib import Path

sonar = Sonar()
earlyStopping=EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

logger = logging.getLogger(__name__)

def get_logger(file_name, logging_level,logs_directory):
   
    logs_directory=Path(logs_directory)
    logs_directory.mkdir(exist_ok=True,parents=True)
    logger.setLevel(logging_level)
    fh = logging.FileHandler(logs_directory/file_name, mode='w')
    fh.setLevel(logging_level)
    sh = logging.StreamHandler()
    sh.setLevel(logging_level)
    logging_formatter = logging.Formatter("%(asctime)s:[%(levelname)s]:[%(name)s]:%(message)s")
    fh.setFormatter(logging_formatter)
    sh.setFormatter(logging_formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

class Sentiment(object):

    def __init__(self, data_directory, model_directory, filename, pickle_directory, output_directory, max_length, batch_size, epochs):
        
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.filename = filename
        self.pickle_directory = Path(pickle_directory) 
        self.output_directory = output_directory       
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
    
    def load_data(self): 
        
        logger.debug(f'Load the data {self.filename} from data directory {self.data_directory}')
        data = pd.read_csv(self.data_directory/self.filename)
        indx = data[(data.stars != 5)&(data.stars != 1)].index   
        data = data.drop(indx)
        self.data_sentiment = data.reset_index(drop=True)
          
    def data_prep(self):
        logger.debug('Split the comments into sentences and obtain the vocabulary')
        all_words = []
        toxic_sentences = []
        attitude_list = []
        sentences_list = []
        
        for i in range(len(self.data_sentiment.index)):
            sentences = re.split(r'[.!]',self.data_sentiment['text'][i].replace('\n',''))
            if self.data_sentiment['stars'][i] ==1:
                attitude = 0
            elif self.data_sentiment['stars'][i] ==5:
                attitude = 1
            for sentence in sentences:
                words = list(preprocessing.tokenize(sentence))
                if len(words) >= 3:
                    all_words += words
                    confid_score = sonar.ping(text=sentence)['classes'][1]['confidence']
                    if  confid_score > 0.8:
                        toxic_sentences.append(sentence)
                    attitude_list.append(attitude)
                    sentences_list.append(sentence)
        
        self.all_words = all_words
        self.toxic_sentences = toxic_sentences
        self.attitude_list = attitude_list
        self.sentences_list = sentences_list

    def words_encoder(self):
        logger.debug('Encoding the vocabulary')
    
        self.encoder = LabelEncoder()
        self.encoder.fit(self.all_words)
        label_encoder_dict = defaultdict(LabelEncoder) 
        for key, encoder in label_encoder_dict.items():
            classes = np.array(self.encoder.classes_).tolist()
            bisect.insort_left(classes, 'UNK')
            self.encoder.classes_ = classes
        self.vocab_size = len(self.encoder.classes_)
        
        
    def encoder_transform(self):
        logger.debug('Transfer the words to the coders')
    
        X_sentiment = []
        for sentence in self.sentences_list:
            words = list(preprocessing.tokenize(sentence))
            if len(words) >= 3:
                try:
                    words = words[:self.max_length]
                except:
                    pass
                words_idx = np.array(self.encoder.transform(words))
                arr = np.full(self.max_length, 0)
                arr[:len(words)] = words_idx
                X_sentiment.append(arr)
        self.X_sentiment = np.array(X_sentiment)
        self.X_sentiment_label = np.array(self.attitude_list)
        
    def create_dict(self):
        self.encoder_dict = {'encoder': self.encoder, 'X_sentiment': self.X_sentiment, 'X_sentiment_label': self.X_sentiment_label, 'toxic_sentences': self.toxic_sentences}
        
    def save_pickle(self):
        logger.debug('Save the pickle file')
        with open(self.pickle_directory/'encoder_senti.pickle', 'wb') as handle:
            pickle.dump(self.encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_pickle(self):
        logger.debug('Load the pickle file')
        with open(self.pickle_directory/'encoder_senti.pickle', 'rb') as handle:
            x = pickle.load(handle)
        return x
            
    def train(self):
        logger.debug(f'Building the neural network with epochs as {self.epochs} and batch size as {self.batch_size}')
        encoder_dict = self.load_pickle()
        encoder = encoder_dict['encoder']
        X_sentiment = encoder_dict['X_sentiment']
        X_sentiment_label = encoder_dict['X_sentiment_label']
               
        self.model_sentiment = Sequential()
        self.model_sentiment.add(Embedding(self.vocab_size, 100))
        self.model_sentiment.add(Bidirectional(LSTM(50,dropout = 0.5, recurrent_dropout =0.5)))
        self.model_sentiment.add(Dense(1, activation='sigmoid'))
        self.model_sentiment.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy', f1_score, auc])
        
        self.history_sentiment = self.model_sentiment.fit(X_sentiment,
                                                      X_sentiment_label,
                                                      validation_split=0.1,
                                                      callbacks=[earlyStopping],
                                                      epochs=self.epochs,
                                                      batch_size=self.batch_size,
                                                      class_weight='auto'
                                                      )
    def save_model(self):
        logger.debug(f'Save the model in {self.model_directory}')
        model_json = self.model_sentiment.to_json()
        with open(self.model_directory/'sentiment_model.json', 'w') as json_file:
            json_file.write(model_json)
        self.model_sentiment.save_weights(self.model_directory/'sentiment_model.h5')
        
    def load_model(self):
        logger.debug(f'Load the model from {self.model_directory}')
        json_file = open(self.model_directory/'sentiment_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.loaded_model_sentiment = model_from_json(loaded_model_json)
            self.loaded_model_sentiment.load_weights(self.model_directory/'sentiment_model.h5')
            
        return self.loaded_model_sentiment
        
    def metric_graph(self):
        history_dict = self.history_sentiment.history
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        f1_score = history_dict['f1_score']
        val_f1_score = history_dict['val_f1_score']
        auc = history_dict['auc']
        val_auc = history_dict['val_auc']
        epochs = range(1, len(acc) + 1)
        
        train_set = ['acc', 'loss', 'f1_score', 'auc']
        val_set = ['val_acc', 'val_loss', 'val_f1_score', 'val_auc']
        
        for i in range(len(train_set)):
            logger.debug(f'Generate the figures for {train_set[i]}')
            fig = matplotlib.pyplot.figure()
            matplotlib.pyplot.title(f'Training and validation {train_set[i]}')
            matplotlib.pyplot.xlabel('epochs')
            matplotlib.pyplot.ylabel(f'{train_set[i]}')
            matplotlib.pyplot.plot(epochs, train_set[i], 'r', label=f'{train_set[i]}')
            matplotlib.pyplot.plot(epochs, val_set[i], 'b', label=f'{val_set[i]}')
            matplotlib.pyplot.legend()
            fig.savefig(self.output_directory/f'sentiment_{train_set[i]}.png')    
            
def run(data_directory, model_directory):
    logger.info('Start the sentiment training')
    se = Sentiment(data_directory, model_directory, filename = 'yelp.csv', pickle_directory = 'pickle', output_directory = 'output', max_length = 30, batch_size = 64, epochs = 20)
    se.load_data()
    se.data_prep()
    se.words_encoder()
    se.encoder_transform()
    se.create_dict()
    se.save_pickle()
    se.load_pickle()
    se.train()
    se.save_model()
    try:
        se.metric_graph()  
    except:
        pass  
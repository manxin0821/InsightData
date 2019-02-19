from __future__ import absolute_import, division, print_function
import keras.backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
    Bidirectional, TimeDistributed
from keras.layers.merge import multiply, concatenate
from keras.models import model_from_json, Model, load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import pandas as pd
from util import make_w2v_embeddings, split_and_zero_padding, ManDist, f1_score, auc
from hatesonar import Sonar
import matplotlib.pyplot
import pickle
import logging
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

class Similarity(object):
           
    def __init__(self, data_directory, model_directory, filename, pickle_directory, output_directory, max_seq_length, batch_size, epochs, embedding_dim, n_hidden):
        
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.filename = filename
        self.pickle_directory = Path(pickle_directory)  
        self.output_directory = output_directory      
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.n_epoch = epochs
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        
        
        
    def load_data(self): 
        logger.debug('Load {self.filename} from {self.data_directory}')
        self.data_similarity = pd.read_csv(self.data_directory/self.filename)
               
        
    def data_prep(self):
        logger.debug('Encoding the vocabulary from the data')
        embedding_dict = {}
               
        for q in ['question1', 'question2']:
            self.data_similarity[q + '_n'] = self.data_similarity[q]
        
        self.data_similarity, self.embeddings = make_w2v_embeddings(embedding_dict, self.data_similarity, embedding_dim=self.embedding_dim)
        
        X = self.data_similarity[['question1_n', 'question2_n']]
        Y = self.data_similarity['is_duplicate']
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1)
        X_train = split_and_zero_padding(X_train, self.max_seq_length)
        X_validation = split_and_zero_padding(X_validation, self.max_seq_length)
        
        Y_train = Y_train.values
        Y_validation = Y_validation.values
        
        assert X_train['left'].shape == X_train['right'].shape
        assert len(X_train['left']) == len(Y_train)
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        
   
        
    def create_dict(self):
        self.encoder_dict = {'embeddings': self.embeddings, 'X_train': self.X_train, 'Y_train': self.Y_train, 'X_validation': self.X_validation, 'Y_validation': self.Y_validation}
        
    def save_pickle(self):
        logger.debug('Save the pickle file')
        with open(self.pickle_directory/'encoder_simi.pickle', 'wb') as handle:
            pickle.dump(self.encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_pickle(self):
        logger.debug('Load the pickle file')
        with open(self.pickle_directory/'encoder_simi.pickle', 'rb') as handle:
            x = pickle.load(handle)
        return x
    
    def train(self):
        logger.debug(f'Building the neural network with epochs as {self.n_epoch} and batch size as {self.batch_size}')
        left_input = Input(shape=(self.max_seq_length,), dtype='float32')
        right_input = Input(shape=(self.max_seq_length,), dtype='float32')
        left_sen_representation = self.shared_model(left_input)
        right_sen_representation = self.shared_model(right_input)
        
        man_distance = ManDist()([left_sen_representation, right_sen_representation])
        sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
        similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
        self.model_similarity = Model(inputs=[left_input, right_input], outputs=[similarity])
        
        self.model_similarity.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_score, auc])
        
        self.history_similarity = self.model_similarity.fit([self.X_train['left'], self.X_train['right']], self.Y_train,
                                   batch_size=self.batch_size, callbacks=[earlyStopping], epochs=self.n_epoch,
                                   validation_data=([self.X_validation['left'], self.X_validation['right']], self.Y_validation))
         
        
    def save_model(self):
        logger.debug(f'Save the model in {self.model_directory}')
        model_json = self.model_similarity.to_json()
        with open(self.model_directory/'similarity_model.json', 'w') as json_file:
            json_file.write(model_json)
        self.model_similarity.save_weights(self.model_directory/'similarity_model.h5')
        
    def load_model(self):
        logger.debug(f'Load the model from {self.model_directory}')
        json_file = open(self.model_directory/'similarity_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.loaded_model_similarity = model_from_json(loaded_model_json)
            self.loaded_model_similarity.load_weights(self.model_directory/'similarity_model.h5')
        
        return self.loaded_model_similarity
        
    def shared_model(self, _input):
        logger.debug('Configure the shared bidirectional LSTM')
        embedded = Embedding(len(self.embeddings), self.embedding_dim, weights=[self.embeddings], input_shape=(self.max_seq_length,), trainable=False)(_input)
    
        activations = Bidirectional(LSTM(self.n_hidden, return_sequences=True, dropout = 0.2, recurrent_dropout =0.2), merge_mode='concat')(embedded)
        activations = Bidirectional(LSTM(self.n_hidden, return_sequences=True, dropout = 0.2, recurrent_dropout = 0.2), merge_mode='concat')(activations)
    
        # Attention
        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.n_hidden * 2)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = multiply([activations, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    
        return sent_representation
        
        
    def metric_graph(self):
    
        history_dict = self.history_similarity.history
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
            fig.savefig(self.output_directory/f'similarity_{train_set[i]}.png')  
            
            
def run(data_directory, model_directory):
    logger.info('Start the similarity training')
    si = Similarity(data_directory, model_directory, filename = 'quora.csv', pickle_directory = 'pickle', output_directory = 'output', max_seq_length = 10, batch_size = 1024, epochs = 20, embedding_dim = 300, n_hidden = 50)
    si.load_data()
    si.data_prep()
    si.create_dict()
    si.save_pickle()
    si.load_pickle()
    si.train()
    si.save_model()
    try:
        se.metric_graph()  
    except:
        pass  

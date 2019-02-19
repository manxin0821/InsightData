from sentiment import Sentiment
from similarity import Similarity
from cophi_toolbox import preprocessing
from util import make_w2v_embeddings, split_and_zero_padding
import numpy as np
import pandas as pd
import random
import pickle
import logging
from operator import itemgetter
from pathlib import Path
from hatesonar import Sonar
from keras.utils import CustomObjectScope
from keras.models import model_from_json, load_model
from keras.initializers import glorot_uniform
from pathlib import Path
import bisect
from util import make_w2v_embeddings, split_and_zero_padding, ManDist, f1_score, auc
sonar = Sonar()

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

class Predict(object):
           
    def __init__(self, data_directory, model_directory):
        
        self.data_directory = data_directory
        self.model_directory = model_directory
       
        json_file = open(self.model_directory/'similarity_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.model_similarity = model_from_json(loaded_model_json,custom_objects={'ManDist': ManDist, 'f1_score':f1_score, 'auc':auc})
            self.model_similarity.load_weights(self.model_directory/'similarity_model.h5')
          
        json_file = open(self.model_directory/'sentiment_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.model_sentiment = model_from_json(loaded_model_json,custom_objects={'ManDist': ManDist, 'f1_score':f1_score, 'auc':auc})
            self.model_sentiment.load_weights(self.model_directory/'sentiment_model.h5')
        
        pickle_directory = Path('pickle')
        with open(pickle_directory/'encoder_senti.pickle', 'rb') as handle:
            self.encoder_dict = pickle.load(handle)
        
    def check_se(self, input_sentence):
        encoder = self.encoder_dict['encoder']
        
        words = list(preprocessing.tokenize(input_sentence))
        words = list(map(lambda s: 'unknown' if s not in encoder.classes_ else s, words))
        encoder_classes = encoder.classes_.tolist()
        bisect.insort_left(encoder_classes, 'unknown')
        encoder.classes_ = np.array(encoder_classes)
        word_idx = np.array([encoder.transform(words)])
        
        if self.model_sentiment.predict_proba(word_idx)>0.6:
                senti = 1
        else:
                senti = 0

        return senti
        
    def compare_si(self, senti, input_sentence):
        if senti==0:
            datafile=self.data_directory/'yelp_0.txt'
        elif senti==1:
            datafile=self.data_directory/'yelp_1.txt'
    
        with open(datafile) as f:
            data = f.readlines()
     
        for i in range(len(data)):
            data[i] = data[i].replace('.','').replace('\n','').replace('!','')
    
        result_index=[]
        test_sentence_pairs=[]
        for i in range(len(data)):
            test_sentence=(input_sentence,data[i])
            test_sentence_pairs.append(test_sentence)
        
        embedding_dict = {}
    
        test_df = pd.DataFrame(test_sentence_pairs, columns = ['question1','question2'])
        for q in ['question1', 'question2']:
            test_df[q + '_n'] = test_df[q]
    
        test_df, embeddings = make_w2v_embeddings(embedding_dict, test_df, embedding_dim=300)
    
        X_test = split_and_zero_padding(test_df, 10)
    
        assert X_test['left'].shape == X_test['right'].shape
    
        preds = list(self.model_similarity.predict([X_test['left'], X_test['right']]))
    
        results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
        results.sort(key=itemgetter(2), reverse=True)
    
        return results[0:3]
    
    def pred(self,senti, input_sentence):
        logger.debug('List the detoxicity results')
        class_value = sonar.ping(text=input_sentence)['top_class']
        if class_value == 'offensive_language':
            senti = self.check_se(input_sentence)
            results = self.compare_si(senti, input_sentence)
        else:
            results = 'NO TOXIC'
        return results
        
def run_single(data_directory, model_directory, input_sentence):
    logger.info('Generating results for the input sentence')
    pred_single = Predict(data_directory, model_directory)
    senti = pred_single.check_se(input_sentence)
    ds = pred_single.pred(senti,input_sentence)
    detoxic_sentence = [ds[i][1] for i in range(3)]
    
    print(detoxic_sentence[0])
    print(detoxic_sentence[1])
    print(detoxic_sentence[2])

def run_validation(data_directory, model_directory, output_directory):
    logger.info(f'Generating the validation results in {output_directory}')
    pickle_directory = Path('pickle')

    with open(pickle_directory/'encoder_senti.pickle', 'rb') as handle:
        encoder_dict = pickle.load(handle)

    toxic_sentences = encoder_dict['toxic_sentences']
    toxic_validation = random.sample(toxic_sentences, 20)
    
    detoxic_list = []
    correct_list = []
    sentiment_list = []
    for sentence in toxic_validation:
        logger.debug(f'Processing the sentence: {sentence}')
        pred_validation = Predict(data_directory, model_directory)
        senti = pred_validation.check_se(sentence)
        ds = pred_validation.pred(senti,sentence)
        detoxic_sentence = [ds[i][1] for i in range(3)]
        detoxic_list.append(detoxic_sentence)
        confid_score = [sonar.ping(text=ds[i][1])['classes'][1]['confidence'] for i in range(3)]
        sentiment_score = [pred_validation.check_se(ds[i][1]) for i in range(3)]
        if  all(confid_score[i] < 0.8 for i in range(3)):
            correct_list.append(True)
        else:
            correct_list.append(False) 
        if any(sentiment_score[i] == senti for i in range(3)):
            sentiment_list.append(True)
        else:
            sentiment_list.append(False)
            
    detoxic_df = pd.DataFrame(detoxic_list,columns = ['1st','2nd','3rd'])
    origin_df = pd.DataFrame(toxic_validation, columns=['Origin'])
    correct_df = pd.DataFrame(correct_list,columns=['Detoxicity'])
    sentiment_df = pd.DataFrame(sentiment_list,columns=['Same Sentiment'])
    output_df = pd.concat([origin_df,detoxic_df,sentiment_df,correct_df],axis=1)
    output_df.to_csv(output_directory/'output.csv',index=False)
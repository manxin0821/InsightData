from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
#senti should be 0 or 1, 0 is negative, 1 is positive

def compareSim(input_sentence,modelpath,datapath,senti):
    # data and trained model should be in the same path
    model = load_model(modelpath)
    
    #retrive reviews based on input sentiment
    if senti==0:
    	datafile=datapath+'/yelp_0.txt'
    elif senti==1:
    	datafile=datapath+'/yelp_1.txt'
    
    with open(datafile) as f:
    	datayelp=f.readlines()
    
    #data cleaning 
    for i in range(len(datayelp)):
    	datayelp[i] = datayelp[i].replace('.','').replace('\n','').replace('!','')
    
    # find the top 3 similar reviews
    result_index=[]
    test_sentence_pairs=[]
    for i in range(len(datayelp)):
        test_sentence=(input_sentence,datayelp[i])
        test_sentence_pairs.append(test_sentence)
    
    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])
    preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
    results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
    results.sort(key=itemgetter(2), reverse=True)
    
    # print(results[0][1])
    # print(results[1][1])
    # print(results[2][1])
    
    return results[0:3]
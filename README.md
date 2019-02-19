# deToxic

This project can smartly replace the offensive sentences from the text to the appropriate language without changing the general meanings. 

### Machine Learning Frameworks

The model is built using tensorflow 1.6.0 and Keras 2.1.5. 

## Getting Started

You can train the model with provided datasets and store the results by

```
python scripts/run_proj.py 'training'
```

and validate the model afterwards by

```
python scripts/run_proj.py 'validation'
```

You can also test the model by inputting the customized sentence, for example

```

python scripts/run_proj.py 'test' --input_sentence 'The food here is d**n good'
```


## Reference

* Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM.
* https://github.com/Hironsan/HateSonar
* Siamese Recurrent Architectures for Learning Sentence Similarity. 2016.
* https://github.com/likejazz/Siamese-LSTM
* https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM/

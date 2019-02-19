#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import importlib
import util
importlib.reload(util)
import pickle
from flask import Flask, request, render_template, jsonify
import subprocess
import json
import re

# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__, static_folder='../static', template_folder='../templates')

@app.route('/')
def homepage():
    """Displays the homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def route_predict():

    # senti should be 0 or 1, 0 is negative, 1 is positive

    string_input=request.get_json(force=True)
    input_sentence = string_input['text']
    print(input_sentence)
    #remove_message = "Using TensorFlow backend."
    results = subprocess.check_output(f'python test.py "{input_sentence}"', stderr=subprocess.STDOUT, shell=True)
    results = str(results, 'utf-8').split('\n')[1]
    print(results)
    
    output=[]
    output=results.replace("[","").replace("]","").replace("'","").replace('"',"").replace(",","").split("#")
    print(output[0])
    #pattern = input_sentence + "',(.*?)', " + "array"
    #output = re.findall(pattern, results)
    #output = [x.replace("'", "") for x in output]
    #results=results.split("', '")
    #print(results)
#     output.append(results[0][1])
#     output.append(results[1][1])
#     output.append(results[2][1])

    print(output)
    #return jsonify('a')
    return jsonify(output)
    
    



if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0',port=80)
    app.run(debug=True,  port=9000)





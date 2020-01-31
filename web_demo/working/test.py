
from flask import Flask,request,jsonify
from flask import render_template
import json
import tensorflow as tf
import numpy as np
from  model_run import first_action,record_sample
from hy_config import  train_config

app = Flask(__name__,static_url_path="/static")



@app.route("/")
def index():
    return render_template("index.html")


@app.route('/first', methods=['post'])
def reply():
    req_msg = request.form.to_dict()
    # print(req_msg)
    first_result = first_action(req_msg['data'])
    # print('first_result',first_result)
    return json.dumps({'cut': first_result['cut'],'predict':first_result['predict']})

@app.route('/secondAll', methods=['post'])
def reply_2():
    req_msg = request.form.to_dict()
    print(req_msg['data'])
    record_sample(req_msg['data'])
    return json.dumps({})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
# Flask utils
from flask import Flask,request, render_template,jsonify
from time import time
from datetime import datetime
# models
from bntts.tts import TextToAudio
import soundfile as sf
# Define a flask app
app = Flask(__name__)
# initialize ocr
T2A=TextToAudio("weights")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def consttruct_error(msg,etype,msg_code,details,suggestion=""):
    exec_error={"code":msg_code,
           "type":etype,
           "message":msg,
           "details":details,
           "suggestion":suggestion}
    return exec_error


def update_log(logs):
    with open("logs.log","a+") as log:
        log.write("..............................................\n")
        for k in logs.keys():
            log.write(f"{k}:\t{logs[k]}\n")
        log.write("----------------------------------------------\n")
        


@app.route('/tts', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # container
            logs={}
            logs["req-time"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            save_start=time()
            
            if "text" in request.form['text']:
                text=request.form['text']
                audio=T2A(text)
                sf.write("data/test.wav", audio, 22050)
                logs["file-save-time"]=round(time()-save_start,2)
                update_log(logs)
                return jsonify({"result":"success"})
        
        except Exception as e:
             return jsonify({"error":consttruct_error("","INTERNAL_SERVER_ERROR","500","","please try again with a different text")})
    
    return jsonify({"error":consttruct_error("","INTERNAL_SERVER_ERROR","500","","please try again with a different text")})


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=3031)
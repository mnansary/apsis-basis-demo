#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import os
import pathlib
import base64
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
# Flask utils
from flask import Flask,request, render_template,jsonify
from werkzeug.utils import secure_filename
from time import time
# models
from ocr.ocr import OCR
from ocr.utils import read_img
# Define a flask app
app = Flask(__name__)
# initialize ocr
ocr=OCR()
#res=ocr(img_path,lang="bn")

def convert_and_save(b64_string,file_path):
    with open(file_path, "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))


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
        


@app.route('/ocr', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # container
            logs={}
            logs["req-time"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            req_start=time()
            save_start=time()
            basepath = os.path.dirname(__file__)
            
            if "file" in request.files:
                # Get the file from post request
                f = request.files['file']
                file_path = os.path.join(basepath,"images",secure_filename(f.filename))
                # save file
                file_ext=pathlib.Path(file_path).suffix
                if file_ext not in [".jpg",".png",".jpeg"]:
                    logs["error"]=f"received file-extension:{file_ext}"
                    update_log(logs)
                    return jsonify({"error":consttruct_error("image format not valid.",
                                                            "INVALID_IMAGE","400",
                                                            f"received file-extension:{file_ext}",
                                                            "Please send .png image files")})
                
                f.save(file_path)
                logs["file-name"]=secure_filename(f.filename)    
                
            
            
            logs["file-save-time"]=round(time()-save_start,2)
            
            try:
                img=read_img(file_path)
            except Exception as er:
                logs["error"]="image not readable."
                update_log(logs)
                return jsonify({"error":consttruct_error("image not readable.",
                                                            "INVALID_IMAGE",
                                                            "400",
                                                            "",
                                                            "Please send uncorrupted image files")})
            
                
            
            proc_start=time()
            res=ocr(img)
            logs["ocr-processing-time"]=round(time()-proc_start,2)
            
            update_log(logs)
            #os.remove(file_path)
            
            return jsonify({"result":res})
        
        except Exception as e:
             return jsonify({"error":consttruct_error("","INTERNAL_SERVER_ERROR","500","","please try again with a different image")})
    
    return jsonify({"error":consttruct_error("","INTERNAL_SERVER_ERROR","500","","please try again with a different image")})


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=3030)
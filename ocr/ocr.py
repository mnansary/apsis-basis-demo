#-*- coding: utf-8 -*-
from __future__ import print_function
#-------------------------
# imports
#-------------------------
from .utils import localize_box,LOG_INFO,create_mask
from .detector import Detector
#from .lang import LangClassifier
from .robustScanner import RobustScanner
from paddleocr import PaddleOCR
import os
import cv2
import copy
from tqdm import tqdm
import pandas as pd 
tqdm.pandas()
from time import time
import tensorflow as tf
#-------------------------------------------
# using gpu
# #--------------------------------------------
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
#-------------------------
# class
#------------------------
class OCR(object):
    def __init__(self,weights_dir="weights/"):
        self.line_en=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet',use_gpu=True)
        self.det=Detector()
        LOG_INFO("Loaded Detector and en-rec ")
        self.bnocr=RobustScanner(os.path.join(weights_dir,"rec"))
        LOG_INFO("Loaded bn-rec")
        
        

    def process_boxes(self,img,boxes):
        # boxes
        word_orgs=[]
        for bno in range(len(boxes)):
            tmp_box = copy.deepcopy(boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            word_orgs.append([x1,y1,x2,y2])
        
        # references
        line_refs=[]
        mask=create_mask(img,boxes)
        # Create rectangular structuring element and dilate
        mask=mask*255
        mask=mask.astype("uint8")
        h,w=mask.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//2,1))
        dilate = cv2.dilate(mask, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            line_refs.append([x,y,x+w,y+h])
        line_refs = sorted(line_refs, key=lambda x: (x[1], x[0]))


        # organize       
        data=pd.DataFrame({"words":word_orgs,"word_ids":[i for i in range(len(word_orgs))]})
        # detect line-word
        data["lines"]=data.words.apply(lambda x:localize_box(x,line_refs))
        data["lines"]=data.lines.apply(lambda x:int(x))
        # register as crop
        text_dict=[]
        for line in data.lines.unique():
            ldf=data.loc[data.lines==line]
            _boxes=ldf.words.tolist()
            _bids=ldf.word_ids.tolist()
            _,bids=zip(*sorted(zip(_boxes,_bids),key=lambda x: x[0][0]))
            for idx,bid in enumerate(bids):
                _dict={"line_no":line,"word_no":idx,"crop_id":bid,"box":boxes[bid]}
                text_dict.append(_dict)
        df=pd.DataFrame(text_dict)
        return df

    def __call__(self,img):
        # -----------------------start-----------------------
        result=[]
        # text detection
        word_boxes,crops=self.det.detect(img,self.line_en)
        # process lines
        df=self.process_boxes(img,word_boxes)    
        # ocr
        bn_crops=[crops[idx] for idx in df.crop_id.tolist()]
        bn_text = self.bnocr(bn_crops)
        df["text"]=bn_text
        # format
        for idx in range(len(df)):
            data={}
            data["line_no"]=int(df.iloc[idx,0])
            data["word_no"]=int(df.iloc[idx,1])
            # array 
            poly_res=  []
            poly    =  df.iloc[idx,3]
            for pair in poly:
                _pair=[float(pair[0]),float(pair[1])]
                poly_res.append(_pair)
            
            data["poly"]   =poly_res
            data["text"]   =df.iloc[idx,4]
            result.append(data)
        return result
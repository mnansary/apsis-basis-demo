#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import streamlit as st
import base64
from PIL import Image
import numpy as np
import requests
import pandas as pd
import cv2 
OCR_API="http://192.168.10.110:3030/ocr"
TTS_API="http://192.168.10.110:3031/tts"
#--------------------------------------------------
# main
#--------------------------------------------------
def get_data_url(img_path):
    file_ = open(img_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url


def draw_regions(regions,img):
    mask=np.copy(img)
    for region in regions:
        region = np.array(region).astype(np.int32).reshape((-1))
        region = region.reshape(-1, 2)
        mask = cv2.polylines(mask,[region.reshape((-1, 1, 2))],True,(255, 0, 0),2)
    return mask
        

def main():
    # intro
    st.set_page_config(layout="wide")
    st.title("apsisHOCR: Handwritten text recognition demo")
    
    with st.sidebar:
        st.markdown("**About apsisHOCR**")
        st.markdown("apsisHOCR is a demo ocr system for Handwritten Documents. The full system is build with two components: recognition and detection.")
        st.markdown("---")
        st.markdown(f'<img src="data:image/gif;base64,{get_data_url("apsis.png")}" alt="apsis">'+'   [apsis solutions limited](https://apsissolutions.com/)',unsafe_allow_html=True)
        st.markdown("---")
    
    # For newline
    st.write("\n")
    # Instructions
    st.markdown("*click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1,1,1))
    cols[0].subheader("Input page")
    cols[1].subheader("Detection-output (word level)")
    cols[2].subheader("Recognition Output")
    
    
    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["png", "jpeg", "jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        arr = np.array(image)
        cols[0].image(arr)
        image.save("data/data.png")
        with st.spinner('Executing OCR'):
            res = requests.post(OCR_API, files={'file': open("data/data.png", 'rb')})
            res=res.json()
            # rec
            df=pd.DataFrame(res["result"])
            df=df[['text','line_no','word_no','poly']]
            cols[2].dataframe(df)
            # det
            regions=df.poly.tolist()
            mask=draw_regions(regions,arr)
            cols[1].image(mask)
            # lines
            lines=[]
            for line in df.line_no.unique():
                ldf=df.loc[df.line_no==line]
                ldf.reset_index(drop=True,inplace=True)
                ldf=ldf.sort_values('word_no')
                _ltext=''
                for idx in range(len(ldf)):
                    text=ldf.iloc[idx,0]
                    _ltext+=' '+text
                lines.append(_ltext)
            text="\n".join(lines)
            st.text_area("Multi-line Output", value=text)
            res = requests.post(TTS_API, data={'text':text})
            res=res.json()
            if res["result"]=="success":
                audio_file = open('data/test.wav','rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes,format="audio/wav")
            
                
if __name__ == '__main__':  
    main()
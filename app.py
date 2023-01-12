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

OCR_API="http://192.168.10.110:3030/ocr"
#--------------------------------------------------
# main
#--------------------------------------------------
def get_data_url(img_path):
    file_ = open(img_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url


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
    cols = st.columns((1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Detection-output (word level)")
    
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
        image.save("images/data.png")
        with st.spinner('Executing OCR'):
            with open("images/data.png", 'rb') as f:
                res = requests.post(OCR_API, files={'file': f})
                st.json(res)
            
    # # canvas
    # st.markdown("*draw a word in the canvas*")
    # st.sidebar.header("Configuration")      
    # realtime_update = st.sidebar.checkbox("Update in realtime", True) 
    # # Create a canvas component
    # canvas_result = st_canvas(
    #     fill_color="rgb(255,255,255)",  # Fixed fill color with some opacity
    #     stroke_width=5,
    #     stroke_color="rgb(0,0,0)",
    #     background_color="rgb(255,255,255)",
    #     background_image=None,
    #     update_streamlit=realtime_update,
    #     height=150,
    #     drawing_mode="freedraw",
    #     display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    #     key="main",
    # )
    
    # if st.button("Predict"):
    #     if canvas_result.image_data is not None:
    #         with st.spinner('Loading model...'):
    #             ocr=BHOCR("models/model.h5")
    
    #         with st.spinner('Analyzing...'):
    #             img=np.asarray(canvas_result.image_data).astype(np.uint8)
    #             cv2.imwrite("tests/data.png",img)
    #             st.write("Image saved at:tests/data.png")
    #             img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #             res = pytesseract.image_to_string(img, lang='ben', config='--psm 6')
    #             st.write(f"Tesseract Recognition Before Transformation:",res.split("\n")[0])
    #             text,img=ocr.infer(img)
    #             st.image(img,caption="Grapheme Transformation Result")
    #             st.write(f"Tesseract Recognition After Transformation:",text)
                
    #     else:
    #         st.write("Please Draw a word first!!!")



if __name__ == '__main__':  
    main()
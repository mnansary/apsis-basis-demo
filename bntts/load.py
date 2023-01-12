#-*- coding: utf-8 -*-
"""
@author:Mobassir Hossain
Adaptation:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import os 
import gdown
# link -> hhttps://drive.google.com/drive/folders/1IMCiQpyYBqu98dlRMSINjFNc34fI6zhs?usp=sharing
url = "https://drive.google.com/drive/folders/1IMCiQpyYBqu98dlRMSINjFNc34fI6zhs?usp=sharing"
isExist = os.path.exists('./bangla_tts')
if not isExist:
    gdown.download_folder(url=url, quiet=True, use_cookies=False)   
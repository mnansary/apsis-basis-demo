#-*- coding: utf-8 -*-
"""
@author:Mobassir Hossain
Adaptation:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

from TTS.utils.synthesizer import Synthesizer

test_ckpt   = 'bangla_tts/bn_vits/female/checkpoint_811000.pth'
test_config = 'bangla_tts/bn_vits/female/config.json'
bn_model=Synthesizer(test_ckpt,test_config)

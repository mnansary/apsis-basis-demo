#-*- coding: utf-8 -*-
"""
@author:Mobassir Hossain
"""
from __future__ import print_function
#----------------------
# imports
#----------------------
import re
import os 
import torch
import torchaudio.functional as F
from bnnumerizer import numerize 
from TTS.utils.synthesizer import Synthesizer
from bnunicodenormalizer import Normalizer 
import soundfile as sf

# initialize
bnorm=Normalizer()
def normalize(sen):
    _words = [bnorm(word)['normalized']  for word in sen.split()]
    return " ".join([word for word in _words if word is not None]) 


class TextToAudio(object):
    
    def __init__(self,
                 weights_dir,
                 bn_sample_rate=22050,
                 out_sample_rate=22050,
                 find_nd_replace={
                                  "কেন"  : "কেনো",
                                  "কোন" : "কোনো",
                                  "বল"   : "বলো",
                                  "চল"   : "চলো",
                                  "কর"   : "করো",
                                  "রাখ"   : "রাখো",
                                     },
                 resample_params={"lowpass_filter_width": 64,
                                "rolloff": 0.9475937167399596,
                                "resampling_method": "kaiser_window",
                                "beta": 14.769656459379492}
                ):
        '''
            Instantiates Text to Audio conversion object for bangla
            args:
                weights_dir : path to tts model weights folder with tts superfolder as config and cpkt file
                bn_sample_rate : bangla audio sample rate [optional] default: 22050
                out_sample_rate : audio sample rate [optional] default: 22050
                resample_params : audio resampling parameters [optional]
            resources:
                # Main class: modified from https://github.com/snakers4/silero-models/pull/174
                # Audio converter:https://www.kaggle.com/code/shahruk10/inference-notebook-wav2vec2
                # main:---> https://github.com/mobassir94/comprehensive-bangla-tts/blob/43e2ae8b3f7f862c058da2e22941e01d41ec8ed4/Apps/multilingual_tts_v2.py
        '''
        cpkt            =   os.path.join(weights_dir,"tts","checkpoint_811000.pth")
        config          =   os.path.join(weights_dir,"tts","config.json")
        self.bn_model   =   Synthesizer(cpkt,config)

        self.find_nd_replace=find_nd_replace
        self.bn_sample_rate=bn_sample_rate
        self.sample_rate=out_sample_rate  
        self.resample_params=resample_params
    
    # public
    def bn_tts(self,text):
        '''
            args: 
                text   : bangla text (string)
            returns:
                audio as torch tensor
        '''
        return torch.as_tensor(self.bn_model.tts(text))
    
    
    def exact_replacement(self,text):
        for word,replacement in self.find_nd_replace.items():
            text = re.sub(normalize(word),normalize(replacement),text)
        return text

    def collapse_whitespace(self,text):
        # Regular expression matching whitespace:
        _whitespace_re = re.compile(r"\s+")
        return re.sub(_whitespace_re, " ", text)

    
    def process_text(self,text):
        # numerize
        text=numerize(text)

        # split
        if "।" in text:punct="।"
        else:punct="\n"

        # text blocks
        blocks=text.split(punct)
        blocks=[b for b in blocks if b.strip()]
        
        # create data
        data=[]
        for block in blocks:
            bn_text = block.strip()
            sentenceEnders = re.compile('[।!?]')
            sentences = sentenceEnders.split(str(bn_text))

            for i in range(len(sentences)):
                res = re.sub('\n','',sentences[i])
                res = normalize(res)    
                res = self.collapse_whitespace(res)
                if(len(res)>500):
                    firstpart, secondpart = res[:len(res)//2], res[len(res)//2:]
                    data.append(firstpart)
                    data.append(secondpart)
                else:
                    data.append(res)
        
        return data
    
    def resample_audio(self,audio,sr):
        '''
            resample audio with sample rate
            args:
                audio : torch.tensor audio
                sr: audi sample rate
        '''
        if sr==self.sample_rate:
            return audio
        else:
            return F.resample(audio,sr,self.sample_rate,**self.resample_params)
        
    
    def get_audio(self,data):
        '''
            creates audio from given data 
                * data=List[Tuples(lang,text)]
        '''
        audio_list = []
        for text in data:
            audio=self.bn_tts(text)
            sr=self.bn_sample_rate
            
            if self.resample_audio_to_out_sample_rate:
                audio=self.resample_audio(audio,sr)
                
            audio_list.append(audio)
  
        audio = torch.cat([k for k in audio_list])
        return audio
    
    # call
    def __call__(self,text,resample_audio_to_out_sample_rate=True):
        '''
            args: 
                text   : bangla text (string)
                resample_audio_to_out_sample_rate: for different sample rate in different models, resample the output audio 
                                                   in uniform sample rate 
                                                   * default:True
            returns:
                audio as numpy data
        '''
        self.resample_audio_to_out_sample_rate=resample_audio_to_out_sample_rate
        data=self.process_text(text)
        audio=self.get_audio(data)
        return audio.detach().cpu().numpy()

if __name__=="__main__":
    t2a=TextToAudio("../weights")
    print("loaded model")
    audio=t2a("কথার কথা")
    sf.write("test.wav", audio, 22050)
    print("saved wav")
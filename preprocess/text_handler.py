import re 
import string
import numpy as np 
from bs4 import BeautifulSoup
import unicodedata 

patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}


def remove_numeric(text): 
    return ''.join(c for c in text if c not in "1234567890")

def lower_text(text):
    return ''.join(c.lower() for c in text)
    
def remove_html(raw_html):
    cleantext = BeautifulSoup(raw_html, "html.parser").text
    return cleantext 

def remove_multiple_space(text):
    return re.sub("\s\s+", " ", text)

def remove_punctuation(text):
    text = re.sub(r"''"," ",text)
    text=re.sub(r"[''-()#/@;:<>{}`/+=~|..!?[]]", " ", text)
    text=re.sub("\s\s+", " ", text)
    text = text.strip()
    return text

def extract_emojis(str):
    return [c for c in str if c in emoji.UNICODE_EMOJI]

def remove_tone(text):
    for regex, replace in patterns.items():
        text = re.sub(regex, replace, text)
        # upper case
        text = re.sub(regex.upper(), replace.upper(), text)
    return text


def remove_null_element(text):
    text = text.replace("''",' ')
    return text






  




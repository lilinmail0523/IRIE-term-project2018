
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import contractions, unicodedata
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim import corpora
from nltk.corpus import stopwords

from gensim.summarization import bm25
import os, re, string, inflect
import pandas as pd

cur_dir = os.getcwd()


# clean data reference: https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html



class clean_data():


    #remove_between_square_brackets
    def denoise_text(text):
        text = re.sub('\[[^]]*\]', '', text)
        return text
    #remove contraction like I'm don't
    def replace_contractions(text):
        return contractions.fix(text)

    #remove non ascii
    def remove_non_ascii(words):
        output = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            output.append(new_word)
        return output

    #A => a
    def to_lowercase(words):
        output = []
        for word in words:
            new_word = word.lower()
            output.append(new_word)
        return output

    #remove_punctuation reference : https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    def remove_punctuation(words):
        translate_table = dict((ord(char), None) for char in string.punctuation)
        output = words.translate(translate_table)
        return output

    #replace number 123.... to one two three...
    def replace_numbers(words):
        p = inflect.engine()
        output = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                output.append(new_word)
            else:
                output.append(word)
        return output

    #remove_stopwords
    def remove_stopwords(words):
        output = []
        for word in words:
            if word not in stopwords.words('english'):
                output.append(word)
        return output

    #data preprocessing
    def clean(text):
        if text is not None:
            text = clean_data.denoise_text(text)
            text = clean_data.replace_contractions(text)

            text = clean_data.remove_punctuation(text)


            text = word_tokenize(text)
            text = clean_data.remove_non_ascii(text)
            text = clean_data.to_lowercase(text)
            text = clean_data.replace_numbers(text)
            text = clean_data.remove_stopwords(text)
        return text

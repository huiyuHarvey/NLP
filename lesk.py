# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 20:48:21 2015

Assignment 3 for NLP

@author: Huiyu Yang
"""

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()



def preprocess(string):
    """
    This method is to preprocess the target sentence in string format.
    Tokenize it, remove the function word, and get the stem of each word.
    """
    phrase_words = string.split()
    non_stop_words = [w for w in phrase_words if w not in stopwords.words('english')]
    result = [ps.stem(i) for i in non_stop_words]
    return result
    
def count_overlap(ambiguous_word_dict, rest_words_dict):
    '''
    1. Count the frequency of each word in rest_word_dict.
    Make sure that each word in one sense would only count once.
    2. Compare the each word in ambiguous_word_dict with words in
    the word_frequency, count the overlap.
    3. Return the sense with most overlapping words.
    '''
    
    overlaps = 0
    max_overlap = 0
    lesk_sense = None
    
    word_frequency = {}
    for sense in rest_words_dict:
        rest_words_dict[sense] = set(rest_words_dict[sense])
        for w in rest_words_dict[sense]:
            if w in word_frequency:
                word_frequency[w] += 1
            else:
                word_frequency[w] = 1
                    
    for sense in ambiguous_word_dict:
        for w in ambiguous_word_dict[sense]:
            if w in word_frequency:
                overlaps += word_frequency[w]
        if overlaps > max_overlap:
            max_overlap = overlaps
            lesk_sense = sense
    return lesk_sense    
    

def original_lesk(target_phrase):
    
    '''
    divide the preprocessed target sentence into 2 parts.
    1st is the ambiguous word, 2nd is the rest words.
    e.g. {[time], [fly, arrow]}
         {[fly], [time, arrow]}
         {[arrow], [time, fly]}
    do it for each word in target sentence.
    '''
    lesk_sense = {}
    for i in range(len(target_phrase)):
        ambiguous_word = target_phrase[i]
        rest_words = target_phrase[:]
        rest_words.remove(ambiguous_word)
        
        #preprocess the senses of ambiguous word.
        #tokenize the dictionary, remove the function words, get the stem of each word.
        ambiguous_word_dict = {}
        for sense in wn.synsets(ambiguous_word):
            sense_tokens = sense.definition().split()
            sense_non_stop = [w for w in sense_tokens if w not in stopwords.words('english')]
            sense_stem = [ps.stem(i) for i in sense_non_stop]
            ambiguous_word_dict[sense] = sense_stem
        
        #preprocess the senses of rest words.
        #tokenize the dictionary, remove the function words, get the stem of each word.
        rest_words_dict = {}
        for w in rest_words:
            for sense in wn.synsets(w):
                sense_tokens = sense.definition().split()
                sense_non_stop = [w for w in sense_tokens if w not in stopwords.words('english')]
                sense_stem = [ps.stem(i) for i in sense_non_stop]
                rest_words_dict[sense] = sense_stem
        
        #use the perdefined the method count_overlap to get the sense with most overlap
        best_sense = count_overlap(ambiguous_word_dict, rest_words_dict)
        for word in target_phrase:
            for ss in wn.synsets(word):
                if ss == best_sense:
                    lesk_sense[best_sense] = ss.definition()
    
    return lesk_sense
    
    
def execute(string):
    pre_result = preprocess(string)
    lesk_sense = original_lesk(pre_result)
    print lesk_sense



if __name__ == '__main__':
    phrase =  "time flies like an arrow."
    execute(phrase)
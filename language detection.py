# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:54:39 2015
python 2.7.8
1st assignment of NLP course. Use perplexity to identify the language of text.
@author: huiyu Yang
"""

import nltk, re, math, random
from nltk.util import ngrams

"""
  The preprocessing function is to read the text file, preprocess it and return a
certain number of words with space between every 2 words.
"""
def preprocessing(training_file):
    #read the file
    raw = training_file.read()
    #use regular expression to define the rule of removing non-alphabetical character
    preprocess_rule = re.compile(r'[^a-zA-Z]')
    #replace the non-alphabetical character with space
    remove_nonalpha = preprocess_rule.sub(' ', raw)
    #lowercase all letters
    lowercase_result = remove_nonalpha.lower()
    #extract all words
    split_tokens = nltk.word_tokenize(lowercase_result)
    #select the needed size of training set
    part_of_tokens = split_tokens[1000:21000]
    #append all seperate tokens with space in between
    preprocessing_result = " ".join(part_of_tokens)
    return preprocessing_result

#count the type of unigram for a certain string and return a frequency dictionary of unigrams.
def count_unigrams(preprocess_string):
    #use the ngram method from nltk to extract all the unigrams into a list
    unigramlist = list(ngrams(preprocess_string, 1))
    #FreqDist method is to count the type of unigrams. Then store the type as key with 
    #corrsponding frequency as value in a dictionary.
    count_unigrams = nltk.FreqDist(unigramlist)
    return count_unigrams

#count the type of bigram for a certain string and return a frequency dictionary of bigrams.
#details would be similar to the count_unigrams function.    
def count_bigrams(preprocess_string):
    bigramlist = list(ngrams(preprocess_string, 2))
    count_bigrams = nltk.FreqDist(bigramlist)
    return count_bigrams

#count the type of trigram for a certain string and return a frequency dictionary of trigrams.
#details would be similar to the count_unigrams function.    
def count_trigrams(preprocess_string):
    trigramlist = list(ngrams(preprocess_string, 3))
    count_trigrams = nltk.FreqDist(trigramlist)
    return count_trigrams
    
"""
this trigram model function takes the string after preprocessing and the frequency dictionary
of unigrams, bigrams and trigrams as input and returns a maximum likelyhood estimation of the
input string in log space. This function is mainly for the training set.
"""
def trigrams_model_mle(training_string, unigrams_fdict, bigrams_fdict, trigrams_fdict): 
    #origianl max likelyhood estimation in log space is 0.
    log_max_likelyhood_estimation = 0
    #this is to compute log(p(w1)), which equals to count(w1)/len(string)
    numerator = unigrams_fdict[(training_string[0],)]
    denominator = len(training_string)
    log_max_likelyhood_estimation += math.log(float(numerator)/denominator, 2)
    #this is to compute log(p(w2/w1)), which equals to count(w1,w2)/count(w1)
    numerator = bigrams_fdict[(training_string[0],training_string[1])]
    denominator = unigrams_fdict[(training_string[0],)]
    log_max_likelyhood_estimation += math.log(float(numerator)/denominator, 2)
    #this is to compute the sum of 
    #log(count(w3,w2,w1)/count(w2,w1))+log(count(w4,w3,w2)/count(w3,w2))+...
    #+log(count(w(n),w(n-1),w(n-2))/count(w(n-1),w(n-2)))
    for i in range(len(training_string)-2):
        sub = training_string[i:i+3]
        numerator = trigrams_fdict[(sub[0], sub[1], sub[2])]
        denominator = bigrams_fdict[(sub[0], sub[1])]
        log_max_likelyhood_estimation += math.log(float(numerator)/denominator, 2)
    #return the maximum likelyhood estimation in log space    
    return log_max_likelyhood_estimation

"""
this trigram model function is similar to the last function, but uses the laplace smoothing
technique for the test data.
"""    
def trigrams_model_laplace(training_string, unigrams_fdict, bigrams_fdict, trigrams_fdict):  
    log_laplace_estimation = 0
    
    numerator = unigrams_fdict[(training_string[0],)]
    denominator = len(training_string)
    log_laplace_estimation += math.log(float(numerator)/denominator, 2)
    
    numerator = bigrams_fdict[(training_string[0],training_string[1])]
    denominator = unigrams_fdict[(training_string[0],)]
    log_laplace_estimation += math.log(float(numerator)/denominator, 2)
    #here use the laplace smoothing technique.
    for i in range(len(training_string)-2):
        sub = training_string[i:i+3]
        numerator = 1 + trigrams_fdict[(sub[0], sub[1], sub[2])]
        #V is the size of bigrams vocabulary
        V = len(bigrams_fdict)
        denominator = V + bigrams_fdict[(sub[0], sub[1])]      
        log_laplace_estimation += math.log(float(numerator)/denominator, 2)
        
    return log_laplace_estimation

"""
This function is to compute perplexity based on the estimation probabilities from
the previous function.
"""
def compute_perplexity(unigram_dict, probability_estimation):
    count_letters = sum(unigram_dict.values())
    perplexity = math.pow(2, (-probability_estimation/count_letters))
    return perplexity
    
"""
This function is to generate random text based on the first 2 given letters and the 
trigrams frequency dictionary.
"""
def generate_random_text(s1, s2, trigrams_fdic):
    
    """
    this inner function is to select the key from a dictionary according to corresponding
    value, which means greater the value, higher chance to select the key.
    """
    def weighted_select(d):
        #generate a random number range from 0 to sum of the value in the dictionary.
        r = random.uniform(0, sum(d.itervalues()))
        #s is the accumulated sum of values
        s = 0.0
        
        for k, w in d.iteritems():
                s += w
                if r < s:
                    return k
        return k
    #generate approximately 300 characters
    loop_count = 300
    #initialize the beginning of the random text
    random_text = s1 + s2
    while loop_count >=1:
        loop_count -= 1
        next_letter = None
        trigram_part_dict = {}
        #search for the trigram with same first 2 letters
        for trigram, v in trigrams_fdic.iteritems():
            if trigram[0] == s1 and trigram[1] == s2:
                trigram_part_dict[trigram[2]] = v
                    
        next_letter = weighted_select(trigram_part_dict)
        random_text += next_letter
        s1, s2 = s2, next_letter
    return random_text
    
    
def execute(lan):
    training_file_name = 'training set ' + lan + '.txt'
    training_file = open(training_file_name)
    preprocessing_result = preprocessing(training_file)
    count1gram = count_unigrams(preprocessing_result)
    count2gram = count_bigrams(preprocessing_result)
    count3gram = count_trigrams(preprocessing_result)
    mle_probability_estimation = trigrams_model_mle(preprocessing_result, count1gram, count2gram, count3gram)
    pp_mle_training = compute_perplexity(count1gram, mle_probability_estimation)
    if lan == 'EN':    
        print 'perplexity of English training set is: ', pp_mle_training
    if lan == 'FR':
        print 'perplexity of French training set is: ', pp_mle_training
    if lan == "NL":
        print 'perplexity of Dutch training set is: ', pp_mle_training
    test_file = open("test data.txt")
    preprocessing_result = preprocessing(test_file)
    laplace_test_pe = trigrams_model_laplace(preprocessing_result, count1gram, count2gram, count3gram)
    pp_laplace_test = compute_perplexity(count1gram, laplace_test_pe)
    print 'perplexity of test set is: ' , pp_laplace_test
    random_text = generate_random_text('c', 'o', count3gram)
    print 'random generated text: \n' + random_text
    
    
    
if __name__ == '__main__':
    for lan in ['EN', 'FR', 'NL']:
        execute(lan)
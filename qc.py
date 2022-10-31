import re
import nltk
import math
import sys
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import time


label_flag = sys.argv[1]
train_file = sys.argv[2]
questions_file = sys.argv[3]

def lowertokens(tokens):
    lowerlist = list()
    for x in tokens:
        lowerlist.append(x.lower())
    return lowerlist

def getcoarse(label):
    return re.findall("^.*:", label)[0][:-1]

def notbothstopwords(tokens):
    tokens_list = nltk.word_tokenize(tokens)
    if(len(tokens_list) == 2):
        if(tokens_list[0] in stop_words and tokens_list[1] in stop_words):
            return False
    return True
    

###########################
# PROCESSAMENTO DE TOKENS #
###########################

question_words = ["how", "why", "where", "when", "who", "what", "do", "while", "which", "define", "name", "whose","whom", "describe"]
stop_words = set(nltk.corpus.stopwords.words("english"))
stop_words = set(filter(lambda x: x not in question_words, stop_words))
stop_words.add('``')
stop_words.add("'s")

stemmer = SnowballStemmer("english", ignore_stopwords=True)

f = open(sys.argv[2], "r")
first_words = set([])
frequency = dict()
labels = dict()
label_frequency = dict()
questions = dict()

for x in f:
    match = re.findall("^.*:", x)
    coarse = re.split(":",match[0])[0]
    
    match = re.findall(":.+\s",x)
    fine = re.split("\s", match[0])[0]
    
    label = coarse+fine
    
    # Creating frequencies and updating them
    if label not in label_frequency:
        label_frequency[label] = 0.0
    label_frequency[label] += 1.0
    
    question = re.split(coarse+fine, x)[1]
    tokens_list = nltk.word_tokenize(question)[:-1]
    tokens = []
    
    # Using islower() to make sure we don't add proper names to the tokens
    for s in range(0, len(tokens_list)):
        if tokens_list[s].islower() or tokens_list[s].lower() in question_words:
            tokens.append(tokens_list[s])

    
    tokens = list(map(lambda x: stemmer.stem(x),tokens))
    n_grams = ngrams(tokens,2)
    n_grams = list(list(n_grams))
    n_grams_list = [ ' '.join(grams) for grams in n_grams]
    tokens = tokens + n_grams_list
    tokens = lowertokens(tokens)
    tokens = list(filter(lambda x: x not in stop_words, tokens))
    tokens = list(filter(lambda x: notbothstopwords(x), tokens))
        
    # Adding tokens to a certain label
    if label not in questions:
        questions[label] = list()
    questions[label].append(lowertokens(tokens))   
    
f.close()


###########################
#      CLASSIFICACAO      #
###########################

dev = open(sys.argv[3])
answers = []
count = 0
current_question = []

t1 = time.time()
for x in dev:
    min = math.inf
    answer_label = str()
    current = 0
	
    # Given the tokens taken from the question, we do ngrams of 2 words
    question_tokens = nltk.word_tokenize(x[:-1])
    current_tokens = []
    
    for s in range(0, len(question_tokens)):
        if question_tokens[s].islower() or question_tokens[s].lower() in question_words:
            current_tokens.append(question_tokens[s].lower())
        
    current_tokens = list(map(lambda x: stemmer.stem(x),current_tokens))
    
    # N_grams of 2 tokens
    n_grams = ngrams(current_tokens,2)
    n_grams = list(n_grams)
    n_grams_list = [ ' '.join(grams) for grams in n_grams]
    current_tokens = current_tokens + n_grams_list
    current_tokens = lowertokens(current_tokens)
    current_tokens = list(filter(lambda x: x not in stop_words, current_tokens))
    current_tokens = list(filter(lambda x: notbothstopwords(x), current_tokens))    
    
    for label in questions:  

        for question in questions[label]:
            
            current = nltk.jaccard_distance(set(current_tokens),set(question))

            if(current < min):
                min = current
                answer_label = label
                current_question = question
                
    answers.append(answer_label)
    
t2 = time.time() 
dev.close()

for x in answers:
    if label_flag == "-coarse":
        print(getcoarse(x))
    else:
        print(x)

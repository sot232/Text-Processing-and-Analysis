#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:47:13 2018
Final Project
@author: jeong-ugim
Purpose : 
By using different ratios
between linguistic elements,
I want to predict what year
book were written.
To do so, here we use
Deep-learning method (Keras)
"""

import re
from requests import get
from bs4 import BeautifulSoup
import nltk
from string import punctuation as punct

"""
If you want to do time sleep,
activate following two lines of code
"""

# from time import sleep
# from random import randint

"""
YOU NEED TO CREATE A FILE CALLED "final_project.txt"
BEFORE YOU RUN THIS CODE.
Perhaps not.
"""

# Variables
result_list = []

url = 'http://www.gutenberg.org/browse/authors/a'
response = get(url)
response.status_code
html_req = response.text

def clean(text):
    clean_r = re.compile('\r')
    clean_n = re.compile('\n')
    clear_r = re.sub(clean_r, '', text)
    clear_n = re.sub(clean_n, '', clear_r)
    return clear_n

def is_noun(pos):
    return pos[:2] == 'NN'


def noun_tr(in_text):
    """
    Compute proper noun-token ratio for input Text.
    is_noun = lambda pos: pos[:2] == 'NN'
        """
    nouns = [word for (word, pos)
             in nltk.pos_tag(in_text)
             if is_noun(pos)]
    return len(nouns) / len(in_text)

def is_verb(pos):
    return pos[:2] == 'VB'


def verb_tr(in_text):
    """
    Compute verb ratio for input text.
    is_verb = lambda pos: pos[:2] == 'VB'
        """
    verbs = [word for (word, pos)
             in nltk.pos_tag(in_text)
             if is_verb(pos)]
    return len(verbs) / len(in_text)


def is_plural_noun(pos):
    return pos[:3] == 'NNS'


def plural_noun_tr(in_text):
    """
    Compute plural noun ratio for input text.
    is_plural_noun = lambda pos: pos[:3] == 'NNS'
        """
    plural_nouns = [word for (word, pos)
                    in nltk.pos_tag(in_text)
                    if is_plural_noun(pos)]
    return len(plural_nouns) / len(in_text)


def is_proper_noun(pos):
    return pos[:3] == 'NNP'


def proper_noun_tr(in_text):
    """
    Compute proper noun ratio for input text.
    is_proper_noun = lambda pos: pos[:3] == 'NNP'
        """
    proper_nouns = [word for (word, pos)
                    in nltk.pos_tag(in_text)
                    if is_proper_noun(pos)]
    return len(proper_nouns) / len(in_text)


def is_adj(pos):
    return pos[:2] == 'JJ'


def adj_tr(in_text):
    """
    Compute adj ratio for input text.
    is_adj = lambda pos: pos[:2] == 'JJ'
        """
    adjs = [word for (word, pos)
            in nltk.pos_tag(in_text)
            if is_adj(pos)]
    return len(adjs) / len(in_text)


def is_particle(pos):
    return pos[:2] == 'RP'


def particle_tr(in_text):
    """
    Compute particle ratio for input text.
    is_particle = lambda pos: pos[:2] == 'RP'
        """
    particles = [word for (word, pos)
                 in nltk.pos_tag(in_text)
                 if is_particle(pos)]
    return len(particles) / len(in_text)


def is_adv(pos):
    return pos[:2] == 'RB'


def adverb_tr(in_text):
    """
    Compute adv ratio for input text.
    is_adv = lambda pos: pos[:2] == 'RB'
        """
    advs = [word for (word, pos)
            in nltk.pos_tag(in_text)
            if is_adv(pos)]
    return len(advs) / len(in_text)


def preposition_tr(in_text):
    regex = r"(?:aboard|about|above|\
    across|after|against|ahead|along|\
    amid|amidst|among|around|as|aside|\
    at|athwart|atop|barring|because|\
    before|behind|below|beneath|beside|\
    besides|between|beyond|but|by|\
    circa|concerning|despite|down|\
    during|except|excluding|following|\
    for|from|in|into|inside|like|minus|\
    near|next|notwithstanding|of|off|\
    on|onto|opposite|out|outside|over|\
    past|plus|regarding|save|since|\
    than|through|till|to|toward|\
    under|underneath|unlike|until|\
    up|upon|versus|via|with|\
    within|without)$"
    preposition_count = len([i for i
                             in in_text
                             if re.match(regex, i, re.I)])
    return preposition_count / len(in_text)


def modal_tr(in_text):
    regex = r'(?:must|can|could|may|might|should|ought)$'
    modal_count = len([i for i
                       in in_text
                       if re.match(regex, i, re.I)])
    return modal_count / len(in_text)


def article_tr(in_text):
    regex = r'(?:the|a|an)$'
    article_count = len([i for i
                         in in_text
                         if re.match(regex, i, re.I)])
    return article_count / len(in_text)


def demonstrative_tr(in_text):
    regex = r'(?:this|that|these|those)$'
    demonstratives_count = len([i for i
                                in in_text
                                if re.match(regex, i, re.I)])
    return demonstratives_count / len(in_text)


def distributive_tr(in_text):
    regex = r'(?:all|both|half|either|neither|each|every)$'
    distributive_count = len([i for i
                              in in_text
                              if re.match(regex, i, re.I)])
    return distributive_count / len(in_text)


def pre_determiner_tr(in_text):
    regex = r'(?:such|rather|quite)$'
    pre_determiner_count = len([i for i
                                in in_text
                                if re.match(regex, i, re.I)])
    return pre_determiner_count / len(in_text)


def quantifier_tr(in_text):
    regex = r'(?:few|little|much|many|lot|most|some|any|enough)$'
    quantifier_count = len([i for i
                            in in_text
                            if re.match(regex, i, re.I)])
    return quantifier_count / len(in_text)


def ttr(in_Text):
    """Compute type-token ratio for input Text.

        in_Text -- nltk.Text object or list of strings
        """
    return len(set(in_Text)) / len(in_Text)


def pro1_tr(in_Text):
    """Compute 1st person pronoun-token ratio for input Text.

        in_Text -- nltk.Text object or list of strings
        """
    regex = r'(?:i|me|my|mine)$'
    pro1_count = len([i for i
                      in in_Text
                      if re.match(regex, i, re.I)])
    return pro1_count / len(in_Text)


def pro2_tr(in_Text):
    """Compute 2nd person pronoun-token ratio for input Text.

        in_Text -- nltk.Text object or list of strings
        """
    regex = r'(?:ye|you(?:rs?)?)$'
    pro2_count = len([i for i
                      in in_Text
                      if re.match(regex, i, re.I)])
    return pro2_count / len(in_Text)


def pro3_tr(in_Text):
    """Compute 3rd person pronoun-token ratio for input Text.

        in_Text -- nltk.Text object or list of strings
        """
    regex = r'(?:he|him|his|she|hers?|its?|they|them|theirs?)$'
    pro3_count = len([i for i
                      in in_Text
                      if re.match(regex, i, re.I)])
    return pro3_count / len(in_Text)


def punct_tr(in_Text):
    """Compute punctuation-token ratio for input Text.

        in_Text -- nltk.Text object or list of strings
        """
    punct_count = len([i for i
                       in in_Text
                       if re.match('[' + punct + ']+$', i)])
    return punct_count / len(in_Text)

txt = """title\tttr\tnoun_tr\tverb_tr\t
plural_noun_tr\tproper_noun_tr
\tadj_tr\tparticle_tr\t
adverb_tr\tpreposition_tr\t
modal_tr\tarticle_tr\t
demonstrative_tr\t
distributive_tr\t
pre_determiner_tr\t
quantifier_tr\t
pro1_tr\tpro2_tr\t
pro3_tr\tpunct_tr\texpectedDate\n"""
clear_n = re.compile('\n')
clean_n = re.sub(clear_n, '', txt)
with open('final_features.txt', 'a', encoding='utf-8') as out_file:
    out_file.write(clean_n)

"""
This block of code extracts a data_set
"""

html_soup = BeautifulSoup(html_req, 'html.parser')
# subtract literature links
li_match = html_soup.find_all('li', class_='pgdbetext')
i = 1
for each in li_match:
    # sleep(randint(1, 2))
    expected_date = 0
    english_text = re.search('English', str(each))
    if english_text:
        # We varify this link(each) is English literature
        # Now, we will get each title
        clear_li = re.compile('<li.*?><.*?>')
        clear_li_text = re.sub(clear_li, '', str(each))
        clear_a = re.compile('</a.*')
        title = re.sub(clear_a, '', clear_li_text)
        # Now, we will get the link that will give us plain text
        # text_link includes '"', so let's get rid of it
        text_link = re.search('/ebooks.*?"', str(each), re.S)
        clear_quote = re.compile('"')
        pure_link = re.sub(clear_quote, '', text_link.group())
        link_url = 'http://www.gutenberg.org' + pure_link
        # Then we will get html that will lead us to the actual plain text
        response_link = get(link_url)
        response_link.status_code
        html_link_req = response_link.text
        plain_text_re = '.*Plain Text UTF-8<.*'
        plain_text_link = re.search(plain_text_re, html_link_req)
        # Then we will get the actual plain text
        search_url = re.search('ebooks/.*?.txt', plain_text_link.group())
        clear_ebooks = re.compile('ebooks/')
        if search_url.group():
            clear_front = re.sub(clear_ebooks, '', search_url.group())
            clear_txt = re.compile('.txt')
            clear_url = re.sub(clear_txt, '', clear_front)
            text_url = 'http://www.gutenberg.org/cache/epub/' + clear_url + '/pg' + clear_url + '.txt'
            response_text = get(text_url)
            response_text.status_code
            html_text = response_text.text
            # We don't want to store this text into a variable.
            # If we do, big-oh will be too high.
            raw_text = clean(html_text)
            # Let's get our expected date
            date_line = re.search('Release Date.*', raw_text)
            clear_front = re.compile('Release Date.*?,.')
            clear_back = re.compile('..EBook.*')
            if date_line.group():
                clean_front = re.sub(clear_front, '', date_line.group())
                clean_back = clean_front[:4]
                #clean_back = re.sub(clear_back, '', clean_front)
                expected_date = int(clean_back)
            tok_text = nltk.word_tokenize(raw_text)
            string = str(ttr(tok_text)) + "\t" + str(noun_tr(tok_text))
            string += "\t" + str(verb_tr(tok_text)) + "\t"
            string += str(plural_noun_tr(tok_text)) + "\t"
            string += str(proper_noun_tr(tok_text)) + "\t"
            string += str(adj_tr(tok_text)) + "\t"
            string += str(particle_tr(tok_text)) + "\t"
            string += str(adverb_tr(tok_text)) + "\t"
            string += str(preposition_tr(tok_text)) + "\t"
            string += str(modal_tr(tok_text)) + "\t"
            string += str(article_tr(tok_text)) + "\t"
            string += str(demonstrative_tr(tok_text)) + "\t"
            string += str(distributive_tr(tok_text)) + "\t"
            string += str(pre_determiner_tr(tok_text)) + "\t"
            string += str(quantifier_tr(tok_text)) + "\t"
            string += str(pro1_tr(tok_text)) + "\t"
            string += str(pro2_tr(tok_text)) + "\t"
            string += str(pro3_tr(tok_text)) + "\t"
            string += str(punct_tr(tok_text)) + "\t"
            string += str(expected_date) + "\n"
            with open('final_features.txt', 'a', encoding='utf-8') as out_file:
                out_file.write(string)
            with open('final_features.csv', 'a', encoding='utf-8') as out_file:
                print(ttr(tok_text), noun_tr(tok_text),
                      verb_tr(tok_text), plural_noun_tr(tok_text),
                      proper_noun_tr(tok_text), adj_tr(tok_text),
                      particle_tr(tok_text), adverb_tr(tok_text),
                      preposition_tr(tok_text), modal_tr(tok_text),
                      article_tr(tok_text), demonstrative_tr(tok_text),
                      distributive_tr(tok_text),
                      pre_determiner_tr(tok_text), quantifier_tr(tok_text),
                      pro1_tr(tok_text), pro2_tr(tok_text),
                      pro3_tr(tok_text), punct_tr(tok_text),
                      str(expected_date),
                      sep=',', file=out_file)            
    i+=1
    # optional two lines
    if i > 15:
        break

"""
Deep-Learning Part
BLACK BOX
"""

import numpy as np
from numpy import genfromtxt
#import csv
from keras.models import Sequential
from keras.layers import Dense
from math import floor
import matplotlib.pyplot as plt

with open('final_features.csv', encoding='utf-8') as csv_file:
    dataSet = genfromtxt('final_features.csv', delimiter=',')

"""
X : index 0-18
y : index 19
"""

X = []
y = []

m = dataSet.shape[0]
for i in range(m):
    X.append(dataSet[i][0:18])
    y.append(dataSet[i][19])

X = np.array(X)
y = np.array(y)

"""
Split data into training and testing sets
80% for the tranining set and 20% for the testing sets
"""

splitRatio = 0.8
index = floor(X.shape[0]*splitRatio)
xList = np.split(X, [index])
yList = np.split(y, [index])

X_train = xList[0]
X_test = xList[1]
y_train = yList[0]
y_test = yList[1]

#Cast python lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

"""
Construct our neural network
"""

model = Sequential()

"""
FIRST TRY
"""

"""
#some model hyperparameters
modelHeight = 64
inputSize = 18
numHiddenLayers = 16
outputSize = 1

model.add(Dense(units=modelHeight, 
                activation='relu',
                input_shape=(inputSize,)))
for i in range(numHiddenLayers - 1):
    model.add(Dense(units=modelHeight, 
                    activation='relu'))                   
model.add(Dense(units=outputSize, 
                activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
              
history = model.fit(X_train, y_train, epochs=150, batch_size=10)
"""

"""
SECOND TRY
"""

model.add(Dense(12, input_dim=18, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X, y, epochs=250, batch_size=20)

#Plot results
totalAcc = history.history['acc']
plt.figure(figsize=(10,10))
plt.plot(totalAcc, label="Batch Size: 20")
plt.title("Classification Accuracy Of NN\n")
plt.xlabel("Epoch Step")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

"""
Things you can add
"""

"""
domains = ['a', 'b', 'c', 'd', 'e', 'f', 'g'
           'h', 'i', 'j', 'k', 'l', 'm', 'n',
           'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z']

url = 'http://www.gutenberg.org/browse/authors/' + str()
"""

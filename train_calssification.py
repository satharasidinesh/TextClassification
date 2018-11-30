# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:43:11 2018

@author: Dinesh Satharasi
"""


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

data = pd.read_csv('data\Youtube05-Shakira.csv') 

print(data['CONTENT'][0])
Y = data['CLASS']


text = []
for i in range(370):
    text.append(data['CONTENT'][i])
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
vocabulary = vectorizer.vocabulary_ #dictionary will have all words

X = []
for i in range(370):
    comment = data['CONTENT'][i]
    words = re.split('; |, |\*|\n| ', comment)
    t = []
    for j in range(len(words)):
        lowerWord = words[j].lower()
        if lowerWord in vocabulary:   #some hack 
            t.append(vocabulary[lowerWord])
    X.append(t)

X = np.asarray(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

num_classes = max(Y_train)+1

max_words = 10000  #as we cant pridicat length of comments in advance
tokenizer = Tokenizer(num_words = max_words)

X_train = tokenizer.sequences_to_matrix(X_train, mode = 'freq') #mode = 'binary'
X_test = tokenizer.sequences_to_matrix(X_test, mode = 'freq')

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


model = Sequential()
model.add(Dense(512,input_shape = (max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])


history  = model.fit(X_train,Y_train, batch_size = 32, epochs = 4, verbose = 1, validation_split = 0.1)
score = model.evaluate(X_test, Y_test, batch_size = 32, verbose = 1)
print(score)
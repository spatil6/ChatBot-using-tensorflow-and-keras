# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 07:37:26 2017

@author: Sandip
"""

import os
import json
import nltk
import gensim
import numpy as np
from scipy import spatial
from gensim import corpora, models, similarities
import pickle
from keras.models import Sequential
from keras.layers.recurrent import LSTM,SimpleRNN
import tensorflow
from keras.models import load_model
from sklearn.model_selection import train_test_split

os.chdir("D:\ChatBot");
# word2vec pre trained model download it from https://github.com/jhlau/doc2vec
model = gensim.models.Word2Vec.load('word2vec.bin');
path2="corpus";
#Sample conversation data
file=open(path2+'/conversation.json');
data = json.load(file)
cor=data["conversations"];

x=[]
y=[]
#Creating two list which contains request and response
for i in range(len(cor)):
    for j in range(len(cor[i])):
        if j<len(cor[i])-1:
            x.append(cor[i][j])
            y.append(cor[i][j+1])
tok_x=[]
tok_y=[]
#tokenize request & response list using nltk
nltk.download('punkt')
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))

sentend=np.ones((300,),dtype=np.float32)

#Creation of vector
vec_x=[]
for sent in tok_x:
    sentvec=[model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
vec_y=[]
for sent in tok_y:
    sentvec=[model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)

#Limiting sentece size of sentence to 14
for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
for tok_sent in vec_x:
    if len(tok_sent) < 15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
for tok_sent in vec_y:
    if len(tok_sent) < 15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)
#Saving vectors to local file
with open('conversation.pickle','wb') as f:
    pickle.dump([vec_x,vec_y],f)
type(vec_x)
#Convert list to numpy array
vec_x=np.array(vec_x,dtype=np.float64)
vec_y=np.array(vec_y,dtype=np.float64)
# Create training and validation dataset
x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)
# create sequential model
model=Sequential()
#add layers to model
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
#compile model with loss function cosine proximity
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])
#Train model
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
#Save model on local file sytem for reuse
model.save('LSTM1.h5')
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM1.h10')
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM1.h15')
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM1000.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM1500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM2000.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM2500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM3000.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM3500.h5');
 
#Loading model from local file system, below two steps are optional if trained model is already in memory
model=load_model('LSTM5000.h5')
mod = gensim.models.Word2Vec.load('word2vec.bin');
#Run application on python console
while(True):
    x=input("")
  #  print(x)
    if x in ['exit']:
        break
    sent=nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.vocab]
    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend)
    sentvec=np.array([sentvec])
    #Prediction
    result=model.predict(sentvec)
    
    #print(mod.most_similar([result[0][i]])[0][0] for i in range(15))
    outputlist=[mod.most_similar([result[0][i]])[0][0] for i in range(15)]
    newlist=[]
    for i in outputlist:
        if i not in ['kleiser','karluah','ballets']:
            newlist.append(i)
    output=' '.join(newlist)
    print(output)
    
#kleiser,karluah,ballets
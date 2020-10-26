#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, tensorflow.keras, os
import pandas as pd, keras, io
import numpy as np
from tensorflow.keras.layers import Dense, Input,Dropout, Embedding, LSTM, Bidirectional, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, AveragePooling1D, TimeDistributed, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score,accuracy_score
from bpe import Encoder
import tensorflow as tf


# In[2]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'


# In[3]:


MAX_WORD_LEN=150 
MAX_NB_WORDS=50000
EMBEDDING_DIM=300
NUM_CLASS=2
PATIENCE = 20
ITERATIONS = 100
BATCH_SIZE = 100
FASTTEXT = '/home/ffajri/Data/Indonesian/cc.id.300.vec'
def load_vectors(fname, word_index):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if word_index.get(tokens[0],-1) != -1:
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def model_with_fasttext(x_train, y_train, x_dev, y_dev, x_test, y_test, tokenizer):
    word_index = tokenizer.word_index
    nb_words = min(MAX_NB_WORDS, len(word_index))
    print('Total words in dict:', nb_words)
    print('Now loading FastText')
    embeddings = load_vectors(FASTTEXT, word_index)
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(-4.2, 4.2, EMBEDDING_DIM)

    # MODEL 
    with tf.device('/gpu:0'):
        embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                                input_length=MAX_WORD_LEN, trainable=False)
        
        tweet = Input(shape=(MAX_WORD_LEN,), dtype='int32')
        embedded_sequences = embedding_layer(tweet)

        lstm_cell = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=False, dropout=0.3, recurrent_dropout=0.3)
        doc_vector = Bidirectional(lstm_cell, merge_mode='concat')(embedded_sequences)
        
        
        sign = Dense(NUM_CLASS, activation='softmax')(doc_vector)
        sent_model = Model([tweet], [sign])
        sent_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        bestf1=0.0; patience = 0
        for i in range(ITERATIONS):
            if patience is PATIENCE:
                break
            sent_model.fit([x_train], [y_train], batch_size=BATCH_SIZE, 
                       epochs=1, shuffle=True, verbose=False)
            prediction=sent_model.predict([x_dev], batch_size=1000)
            predicted_label = np.argmax(prediction,axis=1)
            f1score = f1_score(y_dev,predicted_label)
            if f1score > bestf1:
                print('Epoch ' + str(i) +' with dev f1: '+ str(f1score))
                bestf1 = f1score
                sent_model.save('save_w_fasttext.keras')
                patience = 0
            else:
                patience += 1
        sent_model = load_model('save_w_fasttext.keras')
        prediction=sent_model.predict([x_test], batch_size=1000)
        predicted_label = np.argmax(prediction,axis=1)
    f1score = f1_score(y_test,predicted_label)
    print('Test F1: ',f1score)
    print('-----------------------------------------------------------------------------------')
    return f1score


def model(x_train, y_train, x_dev, y_dev, x_test, y_test, tokenizer):
    word_index = tokenizer.word_index
    nb_words = min(MAX_NB_WORDS, len(word_index))
    print('Total words in dict:', nb_words)
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_matrix[i] = np.random.normal(-4.2, 4.2, EMBEDDING_DIM)

    embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                                input_length=MAX_WORD_LEN, trainable=False)
    
    # MODEL 
    with tf.device('/gpu:0'):
        tweet = Input(shape=(MAX_WORD_LEN,), dtype='int32')
        embedded_sequences = embedding_layer(tweet)

        lstm_cell = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=False)
        doc_vector = Bidirectional(lstm_cell, merge_mode='concat')(embedded_sequences)

        sign = Dense(NUM_CLASS, activation='softmax')(doc_vector)
        sent_model = Model([tweet], [sign])
        sent_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        bestf1=0.0; patience = 0
        for i in range(ITERATIONS):
            if patience is PATIENCE:
                break
            sent_model.fit([x_train], [y_train], batch_size=BATCH_SIZE, 
                       epochs=1, shuffle=True, verbose=False)
            prediction=sent_model.predict([x_dev], batch_size=1000)
            predicted_label = np.argmax(prediction,axis=1)
            f1score = f1_score(y_dev,predicted_label)
            if f1score > bestf1:
                print('Epoch ' + str(i) +' with dev f1: '+ str(f1score))
                bestf1 = f1score
                sent_model.save('save.keras')
                patience = 0
            else:
                patience += 1
        sent_model = load_model('save.keras')
        prediction=sent_model.predict([x_test], batch_size=1000)
        predicted_label = np.argmax(prediction,axis=1)
    f1score = f1_score(y_test,predicted_label)
    print('Test F1: ',f1score)
    print('-----------------------------------------------------------------------------------')
    return f1score

def train_and_test_fasttext(x_train, y_train, x_dev, y_dev, x_test, y_test):    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    tokenizer.fit_on_texts(x_train)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    
    max_len = max([len(t) for t in x_train])
    print ('Max Len', max_len)
    max_len = MAX_WORD_LEN
    x_train = sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_len, padding='post')
    x_dev = sequence.pad_sequences(x_dev, maxlen=max_len, padding='post')
    return model_with_fasttext(x_train, to_categorical(y_train), x_dev, y_dev, x_test, y_test, tokenizer)

def train_and_test(x_train, y_train, x_dev, y_dev, x_test, y_test):   
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_train)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    
    max_len = max([len(t) for t in x_train])
    print ('Max Len', max_len)
    #assert max_len < MAX_WORD_LEN
    max_len = MAX_WORD_LEN
    x_train = sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_len, padding='post')
    x_dev = sequence.pad_sequences(x_dev, maxlen=max_len, padding='post')
    return model(x_train, to_categorical(y_train), x_dev, y_dev, x_test, y_test, tokenizer)


# In[ ]:


print('Train with normal random embedding')
print('Batch Size', BATCH_SIZE)
f1s = 0.0
for idx in range(5):
    train = pd.read_excel('data/folds/train'+str(idx)+'.xlsx')
    dev = pd.read_excel('data/folds/dev'+str(idx)+'.xlsx')
    test = pd.read_excel('data/folds/test'+str(idx)+'.xlsx')
    xtrain, ytrain = list(train['minang']), list(train['sentiment'])
    xdev, ydev = list(dev['minang']), list(dev['sentiment'])
    xtest, ytest = list(test['minang']), list(test['sentiment'])
    f1s += train_and_test(xtrain, ytrain, xdev, ydev, xtest, ytest)
print(f1s/5.0)


# In[ ]:


print('Train with FastText Embedding')
print('Batch Size', BATCH_SIZE)
f1s = 0.0
for idx in range(5):
    train = pd.read_excel('data/folds/train'+str(idx)+'.xlsx')
    dev = pd.read_excel('data/folds/dev'+str(idx)+'.xlsx')
    test = pd.read_excel('data/folds/test'+str(idx)+'.xlsx')
    xtrain, ytrain = list(train['minang']), list(train['sentiment'])
    xdev, ydev = list(dev['minang']), list(dev['sentiment'])
    xtest, ytest = list(test['minang']), list(test['sentiment'])
    f1s += train_and_test_fasttext(xtrain, ytrain, xdev, ydev, xtest, ytest)
print(f1s/5.0)


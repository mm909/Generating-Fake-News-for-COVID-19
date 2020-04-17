import pandas as pd
import os
import sys
import time
import keras
import heapq
import numpy as np
import tensorflow as tf
from shutil import copyfile
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, GRU, TimeDistributed, BatchNormalization
from keras.layers import CuDNNLSTM
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback
import random as random
from MLEXPS.MLEXPS import *

df = pd.read_csv("metadata.csv")

# Drop not needed cols
drops = ['cord_uid',
         'sha',
         'source_x',
         'doi',
         'pmcid',
         'pubmed_id',
         'license',
         'abstract',
         'publish_time',
         'authors',
         'journal',
         'Microsoft Academic Paper ID',
         'WHO #Covidence',
         'has_pdf_parse',
         'has_pmc_xml_parse',
         'full_text_file',
         'url']

df = df.drop(drops, axis=1)
df = df.dropna()

data = df.to_numpy()
data = data.flatten()
removenums = []
for i, title in enumerate(data):
    if not title.isascii():
        removenums.append(i)
data = np.delete(data, removenums)

text = ' '
for title in data:
    text += title.lower() + "\n"

removed = ['|', '~', ':', ';', '@', '&', '_', '#', '*', '+', '`', '<', '=', '[', ']']
for char in removed:
    text = text.replace(char,'')

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print('Unique Chars: ', len(chars))
print(chars)

sequenceLength = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - sequenceLength, step):
    sentences.append(text[i: i + sequenceLength])
    next_chars.append(text[i + sequenceLength])
print('Training Examples:', len(sentences))

X = np.zeros((len(sentences), sequenceLength, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print("X.shape:", X.shape)
print("y.shape:", y.shape)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    with open(str(epoch)+".txt", "w") as f:
        f.write('\n')
        f.write('----- Generating text after Epoch: %d\n' % epoch)

        start_index = random.randint(0, len(text) - sequenceLength - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            f.write('\n----- diversity:' + str(diversity) + "\n")

            generated = ''
            sentence = text[start_index: start_index + sequenceLength]
            generated += sentence
            f.write('----- Generating with seed: "' + sentence + '"\n\n')
            f.write(generated)

            for i in range(5000):
                x_pred = np.zeros((1, sequenceLength, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                sentence = sentence[1:] + next_char

                f.write(next_char)
                f.write('')
            f.write('\n')

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model = Sequential()

model.add(CuDNNLSTM(512, return_sequences=True, input_shape=(sequenceLength, len(chars))))
model.add(CuDNNLSTM(256,return_sequences=True))
model.add(CuDNNLSTM(128))
model.add(CuDNNLSTM(64))
model.add(CuDNNLSTM(32))

model.add(Dense(len(chars) * 2))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Dense(len(chars) * 2))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# model.fit(X,y,batch_size=124, epochs=10,shuffle=False,validation_split=0.05,callbacks=[print_callback])

models = [model]
args = [{'x':X,
         'y':y,
         'batch_size':124,
         'epochs':20,
         'shuffle':False,
         'validation_split':0.05,
         'callbacks':[print_callback]}]

ml = MLEXPS()
ml.setTopic('FakeNews')
ml.setCopyFileList(['lstm.py'])
ml.setModels(models)
ml.setArgList(args)
ml.saveBestOnly = False
ml.startExprQ()

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:47:23 2020

Script de Machine Learning 

@author: Silvestre malta
"""
import math
import time
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Flatten, Activation, RNN, SimpleRNN, GRU, Conv1D, MaxPooling1D
# from tensorflow.keras.layers.convolutional import Conv1D
# from tensorflow.keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, plot_model
from gensim.models import Word2Vec
from numpy import array
from numpy import hstack
from time import perf_counter


from openpyxl import load_workbook



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X = list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x= sequences[i:end_ix]
        # print(seq_y)
        X.append(seq_x)

    return X


def word2idx(word):
  return word_model.wv.vocab[word].index

def idx2word(idx):
  return word_model.wv.index2word[idx]


#Função Set Optimizer, Recebe o Optimizer do Excel e retorna aquilo que deve ser
#definido em Model.Compile // Neste momento está definida para tratar Inputs para
#os Optimizers SGD e Adam, pode ser extendida para os restantes

def set_optimizer():

    if optim in optimizer_list:

        if optim == 'SGD' and math.isnan(my_beta_1) == False:

            print('SGD com Beta')
            print(my_beta_1)
            return SGD(lr = my_learning_rate, momentum = my_beta_1)

        elif optim == 'SGD' and math.isnan(my_beta_1) == False:

            print('SGD Sem Beta')
            return SGD(lr = my_learning_rate)

        elif optim == 'Adam' and math.isnan(my_beta_1) == False and math.isnan(my_beta_2) == False:

            print('Adam com Betas')
            print(my_beta_1)
            print(my_beta_2)
            return Adam(lr = my_learning_rate, beta_1 = my_beta_1, beta_2 = my_beta_2)

        else:
            print('Outro Optimizer / Opt Sem Beta', optim)
            return eval(optim)(lr = my_learning_rate)

    else:
        print('Optimizer Inválido!')

#*****************************************************************************





# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-f", "--file", required=True,
	help="Define Dataset Sheet")

ap.add_argument("-t", "--type_of_day", required=True,
	help="0 - all; 1 - without weekend; 2 - without weekend and July, August")

ap.add_argument("-s", "--steps", required=True,
	help="number of steps")

ap.add_argument("-d", "--embeding", required=False,
 	help="define embedding dimensionality")

ap.add_argument("-m", "--hidden_state", required=True,
	help="Hidden state dimensionality")

ap.add_argument("-e", "--epochs", required=True,
	help="Number of Epochs")

ap.add_argument("-b", "--batch", required=True,
	help="Define Batch Size")

ap.add_argument("-l", "--learn_rate", required=True,
	help="Define the Learning Rate")

ap.add_argument("-o", "--optimizer", required=True,
	help="Define the Optimizer")

ap.add_argument("-x", "--beta1", required=False,
	help="Define Beta1")

ap.add_argument("-y", "--beta2", required=False,
	help="Define Beta2")

ap.add_argument("-p", "--my_pacience", required=True,
	help="Define My_Pacience")

ap.add_argument("-v", "--my_model", required=True,
	help="Define Model to be used")

#########

ap.add_argument("-i", "--index", required=False,
	help="Index")

#Set a False ---> ALTERAR

#ap.add_argument("-a", "--activ", required=False,
#	help="Define the Activation Function")

ap.add_argument("-drop", "--dropout", required=False,
	help="Define DropOut")


args = vars(ap.parse_args())

print(args)

print(tf.__version__)







# Model Hyperparameters

# # Embedding dimensionality
# D = 256
# # Hidden state dimensionality was 128
# M = 128
# # Model to use
# my_model = 1
# # All days 1; without weekends 2; without July and August 3;
# my_type_of_days = 1
# use_dropout = True
# my_epochs = 1000
# my_learning_rate=0.1
# my_batch_size = 128
# my_pacience = 100
# my_beta_1 = 0.9


my_file = str(args["file"])
print('Folha Excel:', my_file)

D = int(float(args["embeding"]))
print('PARAMETRO D:', D)

M = int(float(args["hidden_state"]))
print('PARAMETRO M:', M)

use_dropout = True

my_epochs = int(float(args["epochs"]))

my_learning_rate = float(args["learn_rate"])
print('PARAMETRO Learning Rate:', my_learning_rate)

my_batch_size = int(float(args["batch"]))

my_pacience = int(float(args["my_pacience"]))

print('PARAMETRO My Pacience:', my_pacience)

print('PARAMETRO Batch Size:', my_batch_size)


my_beta_1 = float(args["beta1"])
my_beta_2 = float(args["beta2"])

print('BETA1: ', my_beta_1)


my_type_of_days = int(float(args["type_of_day"]))
print("Type of day :", my_type_of_days)

my_model = int(float(args["my_model"]))
print("My Model :", my_model)

optim = args["optimizer"]
print("My Optimizer :", optim)

Dataset = 'Datasets/' + my_file.strip()

print("DATASET :", Dataset)
df = pd.read_excel(Dataset, sheet_name='Folha2')


if my_type_of_days == 2:
    df = df[df['workday'] == 1]
elif my_type_of_days == 3:
    df = df[df['workday'] == 1]
    df = df[df['month'] != 7]
    df = df[df['month'] != 8]

# dataframe.size
size = df.size
linhas = len(df.index)
# dataframe.shape
shape = df.shape

# dataframe.ndim
df_ndim = df.ndim

# printing size and shape
print("Size = {}\nShape ={}\nShape[0] x Shape[1] = {}".
format(size, shape, shape[0]*shape[1]))

last_row = df[-1:]
last_antena=last_row["cell_id"]

print("LAst antenna",last_antena)

#Numero de antenas em cada sequencia
n_steps = 5

X = df["cell_id"]

# em X Passar valores de int para string
X = pd.DataFrame.from_dict(X)
X = X.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')




sentences = split_sequences(X, n_steps)
# print("sentences : ")
# for i in range(10):
#     print(sentences[i])


print('\nTraining word2vec...')
# # word2vec Model #1
# word_model = Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)

# #word2vec Model #2 window = 1
# word_model = Word2Vec(sentences, size=100, min_count=1, window=1, iter=100)

# #word2vec Model #3 window = 1 ; sg=1
# word_model = Word2Vec(sentences, size=100, min_count=1, window=1, iter=100, sg=1)

#word2vec Model #4 window = 1 ; sg=1; hs =1 (hierarquical softmax)
word_model = Word2Vec(sentences, size=100, min_count=1, window=1, iter=100, sg=1, hs=1, negative=0)

pretrained_weights = word_model.wv.syn0
vocab_size, embedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Vocab size :', vocab_size)
print('Embedding Size :', embedding_size)
# print('Checking similar words:')
# for word in ['   3729', '    638', '    670', '  51293']:
#   most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
#   print('  %s -> %s' % (word, most_similar))



# print('\nPreparing the data for LSTM...')
# X = np.zeros([len(sentences), n_steps], dtype=np.int32)
# Y = np.zeros([len(sentences)], dtype=np.int32)
# for i, sentence in enumerate(sentences):
#    for t, word in enumerate(sentence):
#        X[i, t] = word2idx(word)
#    Y[i] = word2idx(sentence[-1])


print('\nPreparing the data for LSTM...')
X = np.zeros([len(sentences), n_steps], dtype=np.int32)
Y = np.zeros([len(sentences)], dtype=np.int32)


for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t] = word2idx(word)
for i, sentence in enumerate(sentences):
    if i != 0 or i != len(sentences):
        Y[i-1] = word2idx(sentence[n_steps-1])
    if i == len(sentences):
        Y[i-1] = word2idx(last_antena)


print('train_x shape:', X.shape)
print('train_y shape:', Y.shape)


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,  shuffle=True)


print('Shape of X train tensor:', X_train.shape)
print('Shape of Y train tensor:', Y_train.shape)
print('Shape of X Test tensor:', X_test.shape)
print('Shape of Y Test tensor:', Y_test.shape)



time_begin = perf_counter()
time_begin_str = time.asctime()

print("**Train Execution Begin :", time_begin_str)

print('\nTraining Model :', my_model)

if my_model == 1:
    print('\n####################################################')
    print('\n####  Training LSTM -- with -- Word2Vec Weights ####')
    print('\n####################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights], name='embedding')(i)
    x = LSTM(M, return_sequences=True, name='LSTM_1')(x)
    if use_dropout:
        x = Dropout(0.2, name='dropout_1')(x)
    x = LSTM(M, return_sequences=True, name='LSTM_2')(x)
    if use_dropout:
        x = Dropout(0.2, name='dropout_2')(x)
    x = TimeDistributed( Dense(vocab_size))(x)
    x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)

elif my_model == 2:
    print('\n#######################################################')
    print('\n####  Training LSTM -- without -- Word2Vec Weights ####')
    print('\n#######################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, name='embedding')(i)
    x = LSTM(M, return_sequences=True, name='LSTM_1')(x)
    if use_dropout:
        x = Dropout(0.2, name='dropout_1')(x)
    x = LSTM(M, return_sequences=True, name='LSTM_2')(x)
    if use_dropout:
        x = Dropout(0.2, name='dropout_2')(x)
    x = TimeDistributed( Dense(vocab_size))(x)
    x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)
    
elif my_model == 3:
    print('\n#######################################################')
    print('\n####  Training CNN+LSTM -- with -- Word2Vec Weights ####')
    print('\n#######################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights], name='embedding')(i)
    x = Conv1D(filters=32, kernel_size=4, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(M, return_sequences=True, name='LSTM_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)
    x = TimeDistributed( Dense(vocab_size))(x)
    x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)
    
elif my_model == 4:
    print('\n#######################################################')
    print('\n####  Training CNN+LSTM -- without -- Word2Vec Weights ####')
    print('\n#######################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, name='embedding')(i)
    x = Conv1D(filters=32, kernel_size=4, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(M, return_sequences=True, name='LSTM_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)
    x = TimeDistributed( Dense(vocab_size))(x)
    x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)
    
elif my_model == 5:
    print('\n#######################################################')
    print('\n####  Training SimpleRNN -- with -- Word2Vec Weights ####')
    print('\n#######################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights], name='embedding')(i)
    x = SimpleRNN(M)(x)
    # x = Dropout(0.2, name='dropout_1')(x)
    # x = TimeDistributed( Dense(vocab_size))(x)
    # x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)
    
elif my_model == 6:
    print('\n#######################################################')
    print('\n####  Training SimpleRNN -- without -- Word2Vec Weights ####')
    print('\n#######################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size,  name='embedding')(i)
    x = SimpleRNN(M)(x)
    # x = Dropout(0.2, name='dropout_1')(x)
    # x = TimeDistributed( Dense(vocab_size))(x)
    # x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)

elif my_model == 7:
    print('\n#######################################################')
    print('\n####  Training GRU -- with -- Word2Vec Weights ####')
    print('\n#######################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights], name='embedding')(i)
    x = GRU(M, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)
    
elif my_model == 8:
    print('\n#######################################################')
    print('\n####  Training GRU -- without -- Word2Vec Weights ####')
    print('\n#######################################################')
    i = Input(shape=(n_steps,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size, name='embedding')(i)
    x = GRU(M, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=vocab_size, activation='softmax', name='softmax')(x)

else:
    print('\n###### I am out of if ######')




model = Model(i, x)
print(model.summary())


# es =  EarlyStopping(monitor='val_loss', patience=my_pacience, mode='min', verbose=1)
# es =  EarlyStopping(monitor='val_loss', patience=my_pacience, verbose=1, mode='auto', min_delta=0.001,
#                     restore_best_weights=True)

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

# callbacks = [
#     EarlyStopping(monitor='val_accuracy', patience=my_pacience, verbose=1, mode='auto', min_delta=0.001,
#                             restore_best_weights=True),
   # EarlyStopping(monitor='val_loss', patience=my_pacience, verbose=1, mode='auto', min_delta=0.001,
   #                          restore_best_weights=True),
                # ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode=max)
               # ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
            # ]

es =  EarlyStopping(monitor='val_accuracy', patience=my_pacience, verbose=1, mode='auto', min_delta=0.001,
                            restore_best_weights=True)

# Compile and fit
model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=set_optimizer(),
    optimizer=SGD(lr = my_learning_rate),
    metrics=['accuracy']
)



print('Training model...')
r = model.fit(
  X_train,
  Y_train,
  batch_size=my_batch_size,
  epochs=my_epochs,
  validation_data=(X_test, Y_test),
  verbose=2,
  callbacks=[es]
)

time_end = perf_counter()
time_end_str = time.asctime()
execution_time = (time_end - time_begin)


print("Execution Time : ", truncate(execution_time,2), "Seconds")
print("Execution Begin :", time_begin_str)
print("Execution End :", time_end_str)
print("Execution Time : ", convert(execution_time))



n_epochs = len(r.history['loss'])
n_epochs = str(n_epochs).zfill(4)

# # evaluate the model
train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy : for Train: %.3f, Test: %.3f' % (train_acc, test_acc))
print('Loss : for Train: %.3f, Test: %.3f' % (train_loss, test_loss))


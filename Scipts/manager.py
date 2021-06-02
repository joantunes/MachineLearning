# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:10:49 2020

@author: Marco & Miguel

Script que lê do dataset e envia os dados necessários (hyperparameters) , para que posteriormente execute o script "W2V-Embed-LSTM-SoftMaxV4.py".
"""

import ray
import math
import time
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
import shlex
import keyboard
import re
import threading
from threading import Thread
from threading import Timer
from subprocess import PIPE, run, Popen
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from time import perf_counter

ray.init(address='10.2.35.43:6379', _redis_password='5241590000000000')

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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

time_begin = perf_counter()


def redifine_rng(dataframe):

    global test_rng

    #if x in (2,3): --> Se quiser tbm no LastTest ?

    if (test_rng.iloc[0]['InitTest'] != 'All' and test_rng.iloc[0]['InitTest'] != 0):

        try:

            txt1 = int(test_rng.iloc[0]['InitTest'])
            txt2 = int(test_rng.iloc[0]['LastTest'])

            if not (1 <= txt1 <= n_testes):
                print('Number out of the range of loaded tests!')
            elif not (1 <= txt2 <= n_testes):
                print('Number out of the range of loaded tests!')
            elif (txt1 > txt2):
                print('Teste 1 > Teste 2 !!!')
            elif (txt1 == 'All' or txt1 == 0):
                new_df = dataframe
                return new_df
                print('All')
            else:
                txt1 = txt1 - 1
                txt2 = txt2

                new_df = dataframe.iloc[txt1:txt2]
                return new_df


        except ValueError:
             print("Input is not Valid!")

        #rng = [int(s) for s in re.findall(r'-?\d+\.?\d*', txt)]
        #print(rng)
    else:
        new_df = dataframe
        return new_df

#--------------------------------------------------------------#

#Get Hyperparameter Dataset

file_location = 'TestsV3.xlsx'

df = pd.read_excel(file_location,sheet_name='Tests', usecols="A:V", skiprows=4)

test_rng = pd.read_excel(file_location,sheet_name='Tests', usecols="A,C", skiprows=1, nrows=1)

df['Activation'] = df['Activation'].str.lower()

#Editar nome das Colunas
df.rename(columns={'N_Steps' : 'n_steps', 'Embed \nDim':'embedding_dim', 'Hidden Units Per layer\nUnits':'hidden_dim'}, inplace=True)

n_testes = len(df.index)

print('##############')

df = redifine_rng(df)

n_testes_rd = len(df.index)

print('Foram Carregados {} de {} Testes dísponíveis!'.format(n_testes_rd, n_testes))
#print('All tests will be executed unless a range is defined. \n To define a range, press R.\n Press Any Other Key will execute all tests!')


#if keyboard.read_key() == "r":

#    df = redifine_rng(df)

print('##############')


def execute(cmd):
    popen = Popen(cmd, stdout=PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()


    #Ler Valores dos Parâmetros de Cada teste
    #Não é recomendável usar iterrows para tratamento de dados


@ray.remote
def comando(a,b,c,d,e,f,g,h,i,j,k,l,m,n,x):
#Executa Script
    command = ['python3', '-u', 'W2V-Embed-LSTM-SoftMaxV4.py', '-s ' + str(a), '-d ' + str(b),\
                '-m ' + str(c), '-l ' + str(d), '-e ' + str(e),\
                '-b ' + str(f), '-f ' + str(g), '-t ' + str(h), '-i ' + str(i),\
                '-o ' + str(j), '-x ' + str(k), '-y ' + str(l),\
                '-p ' + str(m), '-v ' + str(n)]

    for statement in execute(command):
     # print("In the For :", statement)
         print(statement, end="")

    return x

results_ids = []
x = 0
for i, row in df.iterrows():
    num_steps = df.loc[i]['n_steps']
    D = df.loc[i]['Embed_Dim']
    M = df.loc[i]['hidden_dim']
    my_epochs = df.loc[i]['Epochs']
    #activ = df.loc[i]['Activation']
    optim = df.loc[i]['Optimizer']
    learn = df.loc[i]['LR']
    batch = df.loc[i]['Batch_size']
    my_beta1 = df.loc[i]['Beta1']
    my_beta2 = df.loc[i]['Beta2']
    drop = df.loc[i]['Dropout']
    probe = df.loc[i]['Probe']
    dataset = df.loc[i]['DS']
    my_pacience = df.loc[i]['Pacience']
    type_of_day = df.loc[i]['type_of_day']
    my_model = df.loc[i]['Model']
    print("DS :", dataset, "Type Of Day :", type_of_day, " n_steps :", num_steps, " LR :", learn, \
           "Pacience :", my_pacience, "Model :", my_model, "hidden_dim :", M, "My_model : ", my_model)


    results_ids.append(comando._remote(args=[num_steps, D, M, learn, my_epochs, batch, dataset, type_of_day, i, optim, my_beta1, my_beta2, my_pacience, my_model,x], kwargs={}, resources={'node' + str(x): 2}))


    x = x + 1
    if (x == 3):
        x = 0

results = ray.get(results_ids)
#Using universal_newlines=True converts the output to a string
#instead of a byte array.
 # print(command)


    #print(optim)


time_end = perf_counter()
execution_time = (time_end - time_begin)
print("Total Time : ", convert(execution_time))

    #subprocess.call("test.py", shell=True)
    #subprocess.call(['conda activate base', 'somescript.py', somescript_arg1, somescript_val1,...])
    #subprocess.call(['python', 'test.py'])

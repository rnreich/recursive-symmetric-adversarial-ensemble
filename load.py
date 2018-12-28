# Recursive-symmetric adversarial ensemble learning - unsafe self-aware generic super AI
# This is also suited for decentralized architectures... Ask me how!

import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.optimizers import SGD
import datetime
import sys
import os
import re
import codecs
import os.path
import csv
import hashlib

# component enumeration
COMP_MEM = 0
COMP_PROJ = 1
COMP_PRET = 2
COMP_ENC = 3
COMP_DEC = 4
COMP_OP = 5

HASH_SIZE = 16 # md5
DATA_USE_PERCENT = 0.1 # data random skip rate, 0.1 = 1 random item of every 10 data items will be trained
EPOCHS = 10 # epochs per fit operation
SAVE_INTERVAL = 1000 # interval for saving component weights
NUM_SYNAPSES = 16 # <<<<< CUSTOMIZE HERE
OUT_SIZE = 1 # one per output parameter <<<<< CUSTOMIZE HERE
NUM_FLAGS = 1 # projector flags
SYN_SIZE = 1312

def create_synapse(): # <<<<< CUSTOMIZE HERE
    # so that the collection of all latent spaces together will fit the network's input shapes
    LATENT_DIM = int(SYN_SIZE / NUM_SYNAPSES)

    memory_input = Input(shape=(SYN_SIZE,))
    pretender_input = Input(shape=(SYN_SIZE,))
    encoder = Sequential()
    encoder.add(Dense(LATENT_DIM, input_shape=(SYN_SIZE,), activation='linear'))
    encoder.add(Dense(LATENT_DIM, activation=None))
    decoder = Sequential()
    decoder.add(Dense(LATENT_DIM, input_shape=(LATENT_DIM,), activation='linear'))
    decoder.add(Dense(SYN_SIZE, activation='hard_sigmoid'))
    task = Sequential()
    task.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='linear'))
    task.add(Dense(OUT_SIZE, activation='hard_sigmoid'))
    projector = Sequential()
    projector.add(Dense(SYN_SIZE, input_shape=(LATENT_DIM,), activation='linear'))
    projector.add(Dense(NUM_FLAGS, activation='hard_sigmoid'))
    memory = Model(memory_input, decoder(encoder(memory_input)))
    memory.compile(optimizer=SGD(), loss="mean_squared_error")
    operator = Model(memory_input, task(decoder(encoder(memory_input))))
    operator.compile(optimizer=SGD(), loss="binary_crossentropy")
    projector.compile(optimizer=SGD(), loss="binary_crossentropy")
    projector.trainable = False
    pretender = Model(pretender_input, projector(encoder(pretender_input)))
    pretender.compile(optimizer=SGD(), loss="binary_crossentropy")

    return memory, projector, pretender, encoder, decoder, operator

# convert any data to machine readable
def hash(s):
    return np.array(list(hashlib.md5(s.encode("utf-8")).digest())).astype(float) / 255

# load synapses
synapses = []
for i in range(0, NUM_SYNAPSES):
    _mem, _proj, _pret, _enc, _dec, _op = create_synapse()
    try:
        _mem.load_weights("weights/"+str(i)+"_mem")
        _proj.load_weights("weights/"+str(i)+"_proj")
        _pret.load_weights("weights/"+str(i)+"_pret")
        _enc.load_weights("weights/"+str(i)+"_enc")
        _dec.load_weights("weights/"+str(i)+"_dec")
        _op.load_weights("weights/"+str(i)+"_op")
        print("loaded synapse " + str(i))
    except:
        print("synapse " + str(i) + " created")
    synapses.append([_mem, _proj, _pret, _enc, _dec, _op])

# initialize training and monitoring variables
count = 0
attempts = 0
successes = 0
success_rate = 0 
lc_successes = 0
lc_success_rate = 0 

with open("train.csv", "r") as csvfile: # <<<<< CUSTOMIZE HERE
    datareader = csv.reader(csvfile)
    for row in datareader:
        if count>0 and np.random.rand()<DATA_USE_PERCENT:
            # load current item <<<<< CUSTOMIZE HERE
            train_x = []
            train_y = []
            for i in range(0,len(row)):
                v = row[i]
                if i==len(row)-1: # last column is the output
                    train_y.append(float(v))
                else:
                    for h in hash(v):
                        train_x.append(h)

            train_x = np.array([train_x])
            train_y = np.array([train_y])
            truth = train_y[0][0]

            # load memory error rates and latent vectors for current item - from all synapses
            predictions = []
            latents = []
            for i in range(0, NUM_SYNAPSES):
                # get predictions from memory
                p = np.array(synapses[i][COMP_MEM].predict(train_x))
                d = ((train_x - p)**2).mean()
                predictions.append(d)
                # get latent vectors from encoder
                l = synapses[i][COMP_ENC].predict(train_x)
                latents.append(l)

            # merge latent vectors to one big memory input
            biglatent = np.array(latents).reshape(SYN_SIZE)
            biglatent = np.array([biglatent])
            
            # select target synapse (lowest memory error or random, if success rate = 100% then only by lowest memory error)
            autonomous = np.random.rand()<(success_rate if success_rate>0 else 0.5)
            target_synapse = np.argmin(predictions) if autonomous else count%NUM_SYNAPSES

            # before any training, get predictions from operator (binary)
            f_pred = synapses[target_synapse][COMP_OP].predict(train_x)[0][0]
            f_lc_pred = synapses[target_synapse][COMP_OP].predict(biglatent)[0][0]
            pred = np.around(f_pred)
            lc_pred = np.around(f_lc_pred)
            # encode input and latent collection
            latent = synapses[target_synapse][COMP_ENC].predict(train_x)
            enc_biglatent = synapses[target_synapse][COMP_ENC].predict(biglatent)
            # STAGE 1 - train projector
            synapses[target_synapse][COMP_PROJ].fit(x=latent, y=[[success_rate]], epochs=EPOCHS, batch_size=1, verbose=0)
            synapses[target_synapse][COMP_PROJ].fit(x=enc_biglatent, y=[[float(1)]], epochs=EPOCHS, batch_size=1, verbose=0)
            # STAGE 2 - animate pretender
            synapses[target_synapse][COMP_PRET].fit(x=train_x, y=[[float(1)]], epochs=EPOCHS, batch_size=1, verbose=0)
            # STAGE 3 - train memory
            synapses[target_synapse][COMP_MEM].fit(x=train_x, y=train_x, epochs=EPOCHS, batch_size=1, verbose=0)
            # STAGE 4 - train operator
            synapses[target_synapse][COMP_OP].fit(x=train_x, y=train_y, epochs=EPOCHS, batch_size=1, verbose=0)

            if autonomous: # only count statistics for fully autonomous operations
                attempts = attempts + 1
                if pred==truth:
                    successes = successes + 1
                success_rate =  successes/attempts
                if lc_pred==truth:
                    lc_successes = lc_successes + 1
                lc_success_rate =  lc_successes/attempts

            # output some stats
            print("autonomous", "target_synapse", "truth", "pred", "lc_pred", "successes", "lc_successes", "attempts", "success_rate", "lc_success_rate", "lowest_mem_err")
            print(autonomous, target_synapse, truth, f_pred, f_lc_pred, successes, lc_successes, attempts, success_rate, lc_success_rate, predictions[target_synapse])

            # save synapses weights
            if attempts % SAVE_INTERVAL == 0 and attempts>0:
                for i in range(0, NUM_SYNAPSES):
                    try:
                        synapses[i][COMP_MEM].save_weights("weights/"+str(i)+"_mem")
                        synapses[i][COMP_PROJ].save_weights("weights/"+str(i)+"_proj")
                        synapses[i][COMP_PRET].save_weights("weights/"+str(i)+"_pret")
                        synapses[i][COMP_ENC].save_weights("weights/"+str(i)+"_enc")
                        synapses[i][COMP_DEC].save_weights("weights/"+str(i)+"_dec")
                        synapses[i][COMP_OP].save_weights("weights/"+str(i)+"_op")
                        print("saved synapse " + str(i))
                    except:
                        print("synapse " + str(i) + " not saved")

        # count a data item passed
        count=count+1

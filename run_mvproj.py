# Recursive-symmetric adversarial ensemble learning - Unsafe, autonomous self-aware generic super artificial intelligence

import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Activation, Dropout
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
DATA_USE_PERCENT = 0.01 # data random skip rate, 0.1 = 1 random item of every 10 data items will be trained
EPOCHS = 1 # epochs per fit operation
SAVE_INTERVAL = 1000 # interval for saving component weights
IN_SIZE = 82 # input parameters
NUM_SYNAPSES = IN_SIZE
OUT_SIZE = 1 # one per output parameter
NUM_FLAGS = 5 # projector flags - one per measurement
SYN_SIZE = HASH_SIZE * IN_SIZE
# so that the collection of all latent spaces together will fit the network's input shapes
LATENT_DIM = int(SYN_SIZE / NUM_SYNAPSES)

def create_synapse(): 
    memory_input = Input(shape=(SYN_SIZE,))
    pretender_input = Input(shape=(SYN_SIZE,))
    encoder = Sequential()
    # tanh gives twice the space to work with until outputing a result
    encoder.add(Dense(LATENT_DIM, input_shape=(SYN_SIZE,), activation='tanh'))
    # dropout before encoding
    encoder.add(Dropout(1 / NUM_SYNAPSES))
    # sigmoid as in the output
    encoder.add(Dense(LATENT_DIM, activation='sigmoid'))
    decoder = Sequential()
    decoder.add(Dense(LATENT_DIM, input_shape=(LATENT_DIM,), activation='tanh'))
    # dropout before decoding
    decoder.add(Dropout(1 / NUM_SYNAPSES))
    # sigmoid returns a value between 0 and 1
    decoder.add(Dense(SYN_SIZE, activation='sigmoid'))
    task = Sequential()
    # using relu for decisive operation
    task.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='relu'))
    task.add(Dense(OUT_SIZE, activation='sigmoid'))
    projector = Sequential()
    projector.add(Dense(SYN_SIZE, input_shape=(LATENT_DIM,), activation='tanh'))
    projector.add(Dense(NUM_FLAGS, activation='sigmoid'))
    memory = Model(memory_input, decoder(encoder(memory_input)))
    memory.compile(optimizer=SGD(), loss="mean_squared_error")
    operator = Model(memory_input, task(decoder(encoder(memory_input))))
    operator.compile(optimizer=SGD(), loss="mean_squared_error")
    projector.compile(optimizer=SGD(), loss="mean_squared_error")
    projector.trainable = False
    pretender = Model(pretender_input, projector(encoder(pretender_input)))
    pretender.compile(optimizer=SGD(), loss="mean_squared_error")

    return memory, projector, pretender, encoder, decoder, operator

# convert any data to machine readable signals
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
autonomous = True
flags = None
targets = None
latent = None
enc_biglatent = None

with open("train.csv", "r") as csvfile: 
    datareader = csv.reader(csvfile)
    for row in datareader:
        if count>0 and np.random.rand()<DATA_USE_PERCENT:
            # load current item
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

            # select target synapse (lowest memory error or next)
            target_synapse = np.argmin(predictions) if autonomous else count%NUM_SYNAPSES

            # merge latent vectors to one big memory input
            biglatent = np.array(latents).reshape(SYN_SIZE)
            biglatent = np.array([biglatent])
            
            # projector and pretender are trained as a feedback from previous cycle
            if attempts>0:
                # STAGE 1 - train projector
                synapses[target_synapse][COMP_PROJ].fit(x=latent, y=flags, epochs=EPOCHS, batch_size=1, verbose=0)
                synapses[target_synapse][COMP_PROJ].fit(x=enc_biglatent, y=targets, epochs=EPOCHS, batch_size=1, verbose=0)
                # STAGE 2 - animate pretender
                synapses[target_synapse][COMP_PRET].fit(x=biglatent, y=targets, epochs=EPOCHS, batch_size=1, verbose=0)

            # STAGE 3 - train memory
            synapses[target_synapse][COMP_MEM].fit(x=biglatent, y=biglatent, epochs=EPOCHS, batch_size=1, verbose=0)

            # before exposing the network to the correct output, get predictions from operator
            # now it has the adventage of projector/pretender/memory calibration
            f_pred = synapses[target_synapse][COMP_OP].predict(train_x)
            f_lc_pred = synapses[target_synapse][COMP_OP].predict(biglatent)
            pred = np.around(f_pred[0][0])
            lc_pred = np.around(f_lc_pred[0][0])

            # let the projector know how it's currently doing using arbitrary measurements:
            # decoding error, operator error on data, operator error on latents
            # overall success rate with data, overall sucess rate with latents
            # this block must be sent as a feedback for the next training cycle to prevent poisoning
            flags = np.array([predictions[target_synapse], ((truth-f_pred)**2).mean(), ((truth-f_lc_pred)**2).mean(), success_rate, lc_success_rate])
            flags = flags.reshape((1,NUM_FLAGS))
            targets = np.array([float(0),float(0),float(0),float(1),float(1)]).reshape((1,NUM_FLAGS))
            # get target synapse's reaction to the current data - no fitting
            latent = synapses[target_synapse][COMP_ENC].predict(train_x)
            # recursively encoded representation by the target synapse, of all encoded reactions to the current data
            enc_biglatent = synapses[target_synapse][COMP_ENC].predict(biglatent)

            # STAGE 4 - train operator
            synapses[target_synapse][COMP_OP].fit(x=biglatent, y=train_y, epochs=EPOCHS, batch_size=1, verbose=0)

            # negotiator cycle: after a true prediction, the network is rewarded with being able to use it's best synapses
            attempts = attempts + 1
            if pred==truth:
                successes = successes + 1
                autonomous = True
            else:
                autonomous = False
            success_rate =  successes/attempts
            if lc_pred==truth:
                lc_successes = lc_successes + 1
            lc_success_rate =  lc_successes/attempts

            # output some stats
            print("synapse decoder errors: ", predictions)
            print("autonomous: ", autonomous)
            print("target synapse: ", target_synapse)
            print("truth: ", truth)
            print("rounded prediction (for unseen data): ", pred)
            print("predicted value for unseen data: ", f_pred)
            print("predicted value for latent collection: ", f_lc_pred)
            print("successes with unseen data: ", successes)
            print("successes with latent collections: ", lc_successes)
            print("total sessions: ", attempts)
            print("overall proven success rate (unseen data): ", success_rate)
            print("overall proven success rate (latent collections): ", lc_success_rate)
            print("lowest decoding error (unseen data): ", predictions[target_synapse])
            print("projector flags: ", flags)
            print("****************************************************************************")

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

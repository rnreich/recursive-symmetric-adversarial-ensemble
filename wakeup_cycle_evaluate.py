import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adadelta, SGD, RMSprop
import datetime
import sys
import os
import re
import codecs
import os.path
import csv

#--EVALUATION SCRIPT | KNOWLEDGE EXTRACTION MODE | NOT MODIFYING WEIGHT FILES----------#

DATA_FILE = "train.csv"

COMP_MEM = 0
COMP_PROJ_1 = 1
COMP_PROJ_2 = 2
COMP_PRET_1 = 3
COMP_PRET_2 = 4
COMP_GATE_IN = 5
COMP_GATE_OUT = 6
COMP_OP = 7

HASH_SIZE = 8
ALLOW_NON_NUMERIC = True
DATA_USE_PERCENT = 1
EPOCHS_PER_FIT = 1
IN_SIZE = 90
NUM_SYNAPSES = 8
OUT_SIZE = 1
NUM_FLAGS = 1
SYN_SIZE = HASH_SIZE * IN_SIZE
LATENT_DIM = int(SYN_SIZE / NUM_SYNAPSES)

SPRAY_ROUNDS = HASH_SIZE ** 2
GATE_ERROR_DECIMALS = 3

# may find a better one using itself or an external neural network
MEDIAN_SIGNAL = (HASH_SIZE * IN_SIZE * NUM_SYNAPSES * SYN_SIZE * LATENT_DIM)**2

print("synapse size: ", SYN_SIZE)
print("latent size: ", LATENT_DIM)

synapses = []

#------------------------------------------------------------------------------------#

def softhash(s):
    a = np.zeros(HASH_SIZE)
    chars = list(s)

    for x in range(0, len(chars)):
        cx = chars[x]
        sx = str(cx)
        if len(sx)==1:
            ascii_code = ord(cx)
            if ALLOW_NON_NUMERIC or sx.isnumeric():
                wf = ascii_code if not sx.isnumeric() else float(sx) / 10 * 256
                if x >= HASH_SIZE:
                    a[x % HASH_SIZE] = (a[x % HASH_SIZE] + wf) / 2
                else:
                    a[x] = wf

    return a/255

def create_synapse(): 
    memory_input = Input(shape=(SYN_SIZE,))

    pretender1_input = Input(shape=(SYN_SIZE,))
    pretender2_input = Input(shape=(LATENT_DIM,))

    gate_in = Sequential()
    gate_in.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='relu'))
    gate_in.add(Dropout(1-1 / NUM_SYNAPSES))
    gate_in.add(Dense(LATENT_DIM, activation='hard_sigmoid'))

    gate_out = Sequential()
    gate_out.add(Dense(SYN_SIZE, input_shape=(LATENT_DIM,), activation='relu'))
    gate_out.add(Dropout(1-1 / NUM_SYNAPSES))
    gate_out.add(Dense(SYN_SIZE, activation='sigmoid'))

    task = Sequential()
    task.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='relu'))
    task.add(Dense(OUT_SIZE, activation='hard_sigmoid'))

    projector1 = Sequential()
    projector1.add(Dense(SYN_SIZE, input_shape=(LATENT_DIM,), activation='tanh'))
    projector1.add(Dense(NUM_FLAGS, activation='sigmoid'))

    projector2 = Sequential()
    projector2.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='tanh'))
    projector2.add(Dense(NUM_FLAGS, activation='sigmoid'))

    memory = Model(memory_input, gate_out(gate_in(memory_input)))
    memory.compile(optimizer=SGD(), loss="mean_squared_error")

    operator = Model(memory_input, task(gate_out(gate_in(memory_input))))
    operator.compile(optimizer=SGD(), loss="binary_crossentropy")

    projector1.compile(optimizer=Adadelta(), loss="mean_absolute_error")
    projector1.trainable = False

    projector2.compile(optimizer=Adadelta(), loss="mean_absolute_error")
    projector2.trainable = False

    pretender1 = Model(pretender1_input, projector1(gate_in(pretender1_input)))
    pretender1.compile(optimizer=RMSprop(), loss="binary_crossentropy")

    pretender2 = Model(pretender2_input, projector2(gate_out(pretender2_input)))
    pretender2.compile(optimizer=RMSprop(), loss="binary_crossentropy")

    return memory, projector1, projector2, pretender1, pretender2, gate_in, gate_out, operator

def load_synapses():
    for i in range(0, NUM_SYNAPSES):
        _mem, _proj1, _proj2, _pret1, _pret2, _enc, _dec, _op = create_synapse()

        try:
            _mem.load_weights("weights/"+str(i)+"_mem")
            _proj1.load_weights("weights/"+str(i)+"_proj1")
            _proj2.load_weights("weights/"+str(i)+"_proj2")
            _pret1.load_weights("weights/"+str(i)+"_pret1")
            _pret2.load_weights("weights/"+str(i)+"_pret2")
            _enc.load_weights("weights/"+str(i)+"_enc")
            _dec.load_weights("weights/"+str(i)+"_dec")
            _op.load_weights("weights/"+str(i)+"_op")
            print("synapse " + str(i) + " loaded")
        except:
            print("synapse " + str(i) + " does not exit - loading a blank replacement")

        synapses.append([_mem, _proj1, _proj2, _pret1, _pret2, _enc, _dec, _op])

def load_item(row):
    train_x = []
    train_y = []

    for i in range(0,len(row)):
        v = row[i]

        if i == len(row) - 1:
            train_y.append(float(v))
        else:
            shash = softhash(str(v))

            for z in range(0, len(shash)):
                train_x.append(shash[z])

    for i in range(0, IN_SIZE-len(row) + 1):
        zhash = softhash("0")

        for z in range(0, len(zhash)):
            train_x.append(zhash[z])

    train_x = np.array([train_x])
    train_y = np.array([train_y])

    return (train_x, train_y)

#------------------------------------------------------------------------------------#

print("EVALUATION MODE - not modifying weight files")

load_synapses()
intelligence_signal = 0

data_row_index = 0
cycles = np.zeros(NUM_SYNAPSES)
successes = np.zeros(NUM_SYNAPSES)
success_rate = np.zeros(NUM_SYNAPSES)
lc_successes = np.zeros(NUM_SYNAPSES)
lc_success_rate = np.zeros(NUM_SYNAPSES)
streak = np.zeros(NUM_SYNAPSES)
beststreak = np.zeros(NUM_SYNAPSES)

flags = np.array([0]) # no negotiator around
flags = flags.reshape((1,NUM_FLAGS))
targets = np.array([float(1)]).reshape((1, NUM_FLAGS))

# initialization sequence - knowledge extraction mode (sending a median signal)
if True:
    SPRAY_ROUNDS = int(SPRAY_ROUNDS / 2)

    for x in range(0, NUM_SYNAPSES):
        for j in range(0, SPRAY_ROUNDS):
            print("initiating synapse " + str(x) + " | transmitting median signal... ")
            print(MEDIAN_SIGNAL)
            print(str(j+1) + " / " + str(SPRAY_ROUNDS*2) + " rounds")

            syn_rand = np.array([np.ones(SYN_SIZE) / MEDIAN_SIGNAL])
            latent_rand = synapses[x][COMP_GATE_IN].predict(syn_rand)

            synapses[x][COMP_PROJ_1].fit(x=latent_rand, y=flags, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
            synapses[x][COMP_PROJ_2].fit(x=syn_rand, y=flags, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)

        for j in range(0, SPRAY_ROUNDS):
            print("initiating synapse " + str(x) + " | transmitting median signal... ")
            print(MEDIAN_SIGNAL)
            print(str(j+1+SPRAY_ROUNDS) + " / " + str(SPRAY_ROUNDS*2) + " rounds")

            syn_rand = np.array([np.ones(SYN_SIZE) / MEDIAN_SIGNAL])
            latent_rand = synapses[x][COMP_GATE_IN].predict(syn_rand)

            synapses[x][COMP_PRET_1].fit(x=syn_rand, y=targets, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
            synapses[x][COMP_PRET_2].fit(x=latent_rand, y=targets, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)

with open(DATA_FILE, "r") as csvfile: 
    datareader = csv.reader(csvfile)

    for row in datareader:
        if data_row_index>0 and np.random.rand()<DATA_USE_PERCENT:
            train_x, train_y = load_item(row)
            truth = train_y[0][0]
            
            predictions = []
            latents = []

            for i in range(0, NUM_SYNAPSES):
                l = synapses[i][COMP_GATE_IN].predict(train_x)
                latents.append(l)

            biglatent = np.array(latents).reshape(SYN_SIZE)
            biglatent = np.array([biglatent])

            for i in range(0, NUM_SYNAPSES):
                p = np.array(synapses[i][COMP_MEM].predict(biglatent))
                d = ((biglatent - p)**2).mean()
                predictions.append(d)

            predictions = np.array(predictions)
            predictions = np.round(predictions, GATE_ERROR_DECIMALS)
            
            # find the private memory area using human observation
            # using itself for this task is unsafe
            # for safe AI, use some fixed picking method or an external neural network

            print("****************************************************************************")
            print("EVALUATION MODE - not modifying weight files")

            for target_synapse in range(0, NUM_SYNAPSES):
                latent = synapses[target_synapse][COMP_GATE_IN].predict(biglatent)
                dec = synapses[target_synapse][COMP_GATE_OUT].predict(latent)

                f_pred = synapses[target_synapse][COMP_OP].predict(train_x)
                f_lc_pred = synapses[target_synapse][COMP_OP].predict(biglatent)
                
                pred = np.around(f_pred[0][0])
                lc_pred = np.around(f_lc_pred[0][0])

                cycles[target_synapse] = cycles[target_synapse] + 1

                if lc_pred==truth:
                    lc_successes[target_synapse] = lc_successes[target_synapse] + 1
                    streak[target_synapse] = streak[target_synapse] + 1

                    if streak[target_synapse]>beststreak[target_synapse]:
                        beststreak[target_synapse]=streak[target_synapse]
                else:
                    streak[target_synapse] = 0

                if pred==truth:
                    successes[target_synapse] = successes[target_synapse] + 1

                success_rate[target_synapse] =  successes[target_synapse] / cycles[target_synapse]
                lc_success_rate[target_synapse] =  lc_successes[target_synapse] / cycles[target_synapse]

                streak_odds = 2 ** streak[target_synapse]
                beststreak_odds = 2 ** beststreak[target_synapse]

                ssr = (lc_success_rate[target_synapse]) if lc_success_rate[target_synapse] >= 0.5 else lc_success_rate[target_synapse] / 2
                intelligence_signal = (beststreak_odds) / cycles[target_synapse] * ssr
                if intelligence_signal > 1:
                    intelligence_signal = float(1)

                print("----------------------------------------------------------------------------")
                print("data row index: ", data_row_index)
                print("synapse gate errors: ", predictions)
                print("cycle id: ", cycles[target_synapse])
                print("target synapse: ", target_synapse)
                print("truth: ", truth)
                print("prediction (data): ", f_pred)
                print("prediction (latent collection): ", f_lc_pred)
                print("successes (data): ", successes[target_synapse])
                print("successes (latent collections): ", lc_successes[target_synapse])
                print("overall proven success rate (data): ", success_rate[target_synapse])
                print("overall proven success rate (latent collections): ", lc_success_rate[target_synapse])
                print("lowest gate error: ", predictions[target_synapse])
                print("streak: ", streak[target_synapse], " (odds = " + str(streak_odds) + ")")
                print("beststreak: ", beststreak[target_synapse], " (odds = " + str(beststreak_odds) + ")")
                print("intelligence signal: ", intelligence_signal)
                
        data_row_index=data_row_index+1

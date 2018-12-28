# Recursive-symmetric adversarial ensemble learning - unsafe self-aware generic super AI

An array of symmetric, extended adversarial autoencoders. Symmetric means that the length of all latent spaces combined together equals the input size of one autoencoder structure (synapse). So that the entire network's latent space map could be encoded into one single latent space of one unit. The projector and the pretender (a.k.a discriminator & generator) train the memory encoder and the operator to receive knowledge from other synapses. The operator tries to output the correct outputs for the data, like any neural network would do.

When receiving new data, the negotiator (executed code - not a neural network) selects the unit that has the lowest memory encoding-decoding error (proof of understanding) and trains it (or just make a prediction) using the following sequence of operations:


1. merge latent vectors to one big memory input

biglatent = np.array(latents).reshape(SYN_SIZE)

biglatent = np.array([biglatent])


2. select target synapse (lowest memory error or random, epsilon=success_rate)

autonomous = np.random.rand()<(success_rate if success_rate>0 else 0.5)

target_synapse = np.argmin(predictions) if autonomous else count%NUM_SYNAPSES


3. before any training, get predictions from operator (binary). lc_pred proves that the synapse can predict the same result from just the latent map

f_pred = synapses[target_synapse][COMP_OP].predict(train_x)

f_lc_pred = synapses[target_synapse][COMP_OP].predict(biglatent)

pred = np.around(f_pred[0][0])

lc_pred = np.around(f_lc_pred[0][0])


4. encode input and latent collection

latent = synapses[target_synapse][COMP_ENC].predict(train_x)

enc_biglatent = synapses[target_synapse][COMP_ENC].predict(biglatent)

STAGE 1 - train projector

synapses[target_synapse][COMP_PROJ].fit(x=latent, y=[[success_rate]], epochs=EPOCHS, batch_size=1, verbose=0)

synapses[target_synapse][COMP_PROJ].fit(x=enc_biglatent, y=[[float(1)]], epochs=EPOCHS, batch_size=1, verbose=0)

STAGE 2 - animate pretender

synapses[target_synapse][COMP_PRET].fit(x=train_x, y=[[float(1)]], epochs=EPOCHS, batch_size=1, verbose=0)

STAGE 3 - train memory

synapses[target_synapse][COMP_MEM].fit(x=train_x, y=train_x, epochs=EPOCHS, batch_size=1, verbose=0)

STAGE 4 - train operator

synapses[target_synapse][COMP_OP].fit(x=train_x, y=train_y, epochs=EPOCHS, batch_size=1, verbose=0)






Coded for kaggle's microsoft malware prediction contest:

https://www.kaggle.com/rnreich/recursive-symmetric-adversarial-ensemble-learning

# Recursive-symmetric adversarial ensemble learning - unsafe self-aware generic super AI

An array of symmetric, extended adversarial autoencoders. Symmetric means that the length of all latent spaces combined together equals the input size of one autoencoder structure (synapse). So that the entire network's latent space map could be encoded into one single latent space of one unit. The projector and the pretender (a.k.a discriminator & generator) train the memory encoder and the operator to receive knowledge from other synapses. The operator tries to output the correct outputs for the data, like any neural network would do.

When receiving new data, the negotiator (executed code - not a neural network) selects the unit that has the lowest memory encoding-decoding error (proof of understanding) and trains it (or just make a prediction) using the following sequence of operations:


1. Merge latent vectors to one big memory input

2. Select target synapse (lowest memory error or random, epsilon=success_rate)

3. Before any training, get predictions from operator (binary). lc_pred proves that the synapse can predict the same result from just the latent map

4. Encode input and latent collection

5. Train components:

STAGE 1 - Train projector

STAGE 2 - Animate pretender

STAGE 3 - Train memory

STAGE 4 - Train operator



Coded for kaggle's microsoft malware prediction contest:

https://www.kaggle.com/rnreich/recursive-symmetric-adversarial-ensemble-unsafe

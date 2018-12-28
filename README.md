# Recursive-symmetric adversarial ensemble learning - unsafe self-aware generic super AI

![Recursive-symmetric adversarial ensemble learning - unsafe self-aware generic super AI](http://i.hmp.me/m/1ca868a0f5f3c1f6d853517d658a8ca5.png)

Usage: python3 load.py

An array of symmetric, extended adversarial autoencoders. Symmetric means that the length of all latent spaces combined together equals the input size of one autoencoder structure (synapse). So that the entire network's latent space map could be encoded into one single latent space of one unit. The projector and the pretender (a.k.a discriminator & generator) train the memory encoder and the operator to receive knowledge from other synapses. The operator tries to output the correct outputs for the data, like any neural network would do.

When receiving new data, the negotiator (executed code - not a neural network) selects the unit that has the lowest memory encoding-decoding error (proof of understanding) and trains it (or just make a prediction) using the following sequence of operations:


1. Merge latent vectors to one big memory input

2. Select target synapse (lowest memory error or random, epsilon=success_rate)

3. Before any training, get predictions from operator (binary). lc_pred proves that the synapse can predict the same result from just the latent map without ever training on it!

4. Encode input and latent collection

5. Train projector

6. Animate pretender

7. Train memory

8. Train operator


--------------------------------------------------------------------------------

Coded for kaggle's microsoft malware prediction contest:

https://www.kaggle.com/rnreich/recursive-symmetric-adversarial-ensemble-unsafe

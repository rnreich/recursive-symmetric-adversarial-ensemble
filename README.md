## IntelliForge - Recursive-symmetric adversarial ensemble learning
### Indirect high-resolution neural fitting, scalable neural structure operating on raw data ###

![Recursive-symmetric adversarial ensemble learning - Unsafe, autonomous self-aware generic super artificial intelligence](http://i.hmp.me/m/1ca868a0f5f3c1f6d853517d658a8ca5.png)

--------------------------------------------------------------------------------

**Usage (recommended - wakeup cycle): python3 wakeup_cycle.py**

Training an adversarial ensemble lucid dreaming style.

--------------------------------------------------------------------------------

**Usage (tuned version - indirect neural fitting): python3 run_with_labels.py**

- Adaptation to microsoft's presented problem
- Enhanced softhash function

**CHALLENGE #1:** Optimize the intelligence signal in such way, that will train the synapses to always stay above 50% success in predicting. Solution will be posted on 2019/02/01. Winner will receive a hyperparameter optimization solution* for microsoft's contest.

SHA256 of the code that contains the solution:

2ebfc23ab9e0a444b34ad090bc745e6121697bb812cd29c822f8b320e3ede480

**CHALLENGE #2:** Change the code of the negotiator to select synapses by the decoding error of the latent collections instead of the data. Solution will be posted on 2019/02/01. Winner will receive a hyperparameter optimization solution* for microsoft's contest.

SHA256 of the code that contains the solution:

bfff0b07c015d579f9b15210d355af8290a52378fbfc26a13d3e83186ebc27ce

\* Just a copy of my own hyperparameter setup of that time.

Below is a semantic representation of this model, written in IntelliForge syntax, a language that may be used by an adversarial ensemble network to execute dataflow cycles by itself.

Syntax:

***-> Data flow***

***! Fit operation***

***? Predicted data***

--------------------------------------------------------------------------------

NEGOTIATOR INPUT ->

SYNAPSE INPUT ->

BIGLATENT -> MEM! -> BIGLATENT

TRAIN_X -> ENC -> PROJ1! -> TARGETS

TRAIN_X -> ENC -> DEC -> PROJ2! -> TARGETS

TRAIN_X -> PRET1! -> TARGETS

TRAIN_X -> ENC -> PRET2! -> TARGETS


TRAIN_X -> OP -> { "title": "Prediction by softhashed data", "data": ? }

BIGLATENT -> OP -> { "title": "Prediction by latent collection", "data": ? }

TRAIN_X -> ENC -> PROJ1! -> FLAGS

TRAIN_X -> ENC -> DEC -> PROJ2! -> FLAGS

BIGLATENT -> OP! -> TRAIN_Y

--------------------------------------------------------------------------------

**Usage (experimental - indirect neural fitting): python3 run_no_labels.py**

- Not mapping the inputs to any outputs (breakthrough here!)
- Real-time gradient signaling with SGD
- Projectors and pretenders for every layer
- Differential elements - data vs. latent collections
- Multiple hacks for learning underlying patterns
- softhash function

--------------------------------------------------------------------------------

**Usage (experimental - multivariate flags): python3 run_mvproj.py**

- Multivariate arbitrary projector flags
- Optimized hyperparameters
- Intuitive learning - the network learns only from its own reactions to the input data without being exposed to it
- Exploiting the training cycle
- Real-time negotiator cycle improved
- Better monitoring

Please read all comments. Training is slow before the first save interval. The script is set to use only 1% of the data, so it has to be run at least 100 times in order to achieve respectful results.

Things to think about:

- Set the number of projector flags to SYN_SIZE and use it as a communication and action endpoint (interface of the AI), as in deep reinforcement learning, while the "environment" is feeded in real-time into the negotiator input point, which is also compatible with the shape of the projector flags.

- Set the number of projector or operator flags to NUM_SYNAPSES and train it to identify itself in a shuffled input of the network's entire latent map.

- Use various versions of the core synapse model in the same network along with an improved negotiator.

- **Use an unlimited number of synapses from more than one machine, each time selecting the required amount of latent spaces required to fit the input size of the network; The selection of synapses (identifiable) may be made by a neural network under the negotiator's domain (which is safe), or by the network itself (which is unsafe), or both.**

--------------------------------------------------------------------------------

**Usage (basic tutorial): python3 load.py**

An array of symmetric, extended adversarial autoencoders. Symmetric means that the length of all latent spaces combined together equals the input size of one autoencoder structure (synapse). So that the entire network's latent space map could be encoded into one single latent space of one unit. The projector and the pretender (a.k.a discriminator & generator) train the memory encoder and the operator to receive and transmit knowledge to and from other synapses. The operator tries to output the correct outputs for the inputs.

When receiving an input, the negotiator (executed code - not a neural network) selects the unit that has the lowest memory decoding error (proof of understanding) and performs the following sequence of operations:


1. Merge latent vectors from all synapses to one latent map

2. Select target synapse by the lowest memory decoding error of the train_x input (prediction only, no fitting)

3. Before any fitting, get predictions from operator. lc_pred proves that the synapse can predict the same result from the latent map without ever training on latent maps directly. It's also possible to fit the network ONLY on the latent maps. Just change lines 140-144 to:

        synapses[target_synapse][COMP_PRET].fit(x=biglatent, y=[[float(1)]], epochs=EPOCHS, batch_size=1, verbose=0)
        # STAGE 3 - train memory
        synapses[target_synapse][COMP_MEM].fit(x=biglatent, y=biglatent, epochs=EPOCHS, batch_size=1, verbose=0)
        # STAGE 4 - train operator
        synapses[target_synapse][COMP_OP].fit(x=biglatent, y=train_y, epochs=EPOCHS, batch_size=1, verbose=0)

**This is interesting because the network can learn without actually training on any of the input data, only on its own reactions to that data!**

*It's slightly better to train the memory first, see tuned version.*

4. Encode the train_x input and the latent map itself

5. Train projector

6. Animate pretender

7. Train memory

8. Train operator


Note: Activation functions and other hyper-parameters may be adjusted before using this kernel.

--------------------------------------------------------------------------------

Coded for kaggle's microsoft malware prediction contest:

https://www.kaggle.com/rnreich/recursive-symmetric-adversarial-ensemble-unsafe

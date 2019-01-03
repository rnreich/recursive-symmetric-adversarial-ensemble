## IntelliForge - Recursive-symmetric adversarial ensemble learning
### Indirect high-resolution neural fitting, scalable neural structure operating on raw data ###

An assembly of neural networks training together to optimize under a negotiator that fails them deliberately

![Recursive-symmetric adversarial ensemble learning - Unsafe, autonomous self-aware generic super artificial intelligence](http://i.hmp.me/m/1ca868a0f5f3c1f6d853517d658a8ca5.png)

--------------------------------------------------------------------------------

**Usage (recommended - wakeup cycle): python3 wakeup_cycle.py**

Training an adversarial ensemble lucid dreaming style. Wait for the spikes.

To make a prediction (summarized code):

    TARGET_INTELLIGENCE_SIGNAL = 1 # Always 1
    tmp = synapses.copy() # Save a copy for restoration

    # Set the flags and the targets
    flags = np.array([0]) # No negotiator around
    flags = flags.reshape((1,NUM_FLAGS))
    targets = np.array([float(TARGET_INTELLIGENCE_SIGNAL)]).reshape((1, NUM_FLAGS))

    # Prediction loop start

    train_x = load_item(row)
               
    predictions = []
    latents = []

    # Get the entire network's reaction to train_x
    for i in range(0, NUM_SYNAPSES):
        l = synapses[i][COMP_GATE_IN].predict(train_x)
        latents.append(l)

    # Merge them to fit the input size of the synapse
    biglatent = np.array(latents).reshape(SYN_SIZE)
    biglatent = np.array([biglatent])

    # Select the most suited synapse for the data
    for i in range(0, NUM_SYNAPSES):
        p = np.array(synapses[i][COMP_MEM].predict(biglatent))
        d = ((biglatent - p)**2).mean()
        predictions.append(d)

    target_synapse = np.argmin(predictions)

    # Receive the signals from other synapses
    latent = synapses[target_synapse][COMP_GATE_IN].predict(biglatent)
    dec = synapses[target_synapse][COMP_GATE_OUT].predict(latent)

    # Activate the projectors with the flags
    synapses[target_synapse][COMP_PROJ_1].fit(x=latent, y=flags, epochs=EPOCHS_PER_FIT * 2, batch_size=1, verbose=0)
    synapses[target_synapse][COMP_PROJ_2].fit(x=dec, y=flags, epochs=EPOCHS_PER_FIT * 2, batch_size=1, verbose=0)

    # Trigger memory operations
    synapses[target_synapse][COMP_MEM].fit(x=biglatent, y=biglatent, epochs=EPOCHS_PER_FIT * 10, batch_size=1, verbose=0)

    # Activate the projectors with the targets
    synapses[target_synapse][COMP_PROJ_1].fit(x=latent, y=targets, epochs=EPOCHS_PER_FIT * 2, batch_size=1, verbose=0)
    synapses[target_synapse][COMP_PROJ_2].fit(x=dec, y=targets, epochs=EPOCHS_PER_FIT * 2, batch_size=1, verbose=0)
    
    # Activate the pretenders with the targets
    synapses[target_synapse][COMP_PRET_1].fit(x=biglatent, y=targets, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
    synapses[target_synapse][COMP_PRET_2].fit(x=latent, y=targets, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)

    # Make the prediction
    synapses[target_synapse][COMP_OP].predict(biglatent)
    
    # Restore the synapse
    synapses[target_synapse] = tmp[target_synapse]
    
    # Prediction loop end

After making a prediction, don't save the weights, and immediately reload the affected synapse. It's the same as having a one-step trainable copy of the synapse in memory.

**Training**:

The network is forced to learn with high error rates due to the consequences of reaching a clipped 1.0 signal, followed by a wrong prediction. **But this only affects the predictions when the flags are turned on**.

The intelligence signal trasmitted is comprised of a success pattern probability formula, and a differentiation formula containing the success rate of the same network predicting train_x directly, which is unknown to it (line 269):

    intelligence_signal = beststreak_odds / cycles * ssr / (1 - diff) * (1 - success_rate)
    if intelligence_signal > 1:
        intelligence_signal = float(1)

*The variable **ssr** reduces the signal by half if the success rate is under 0.5 - it's easy to observe that the pretenders use it as trick to avoid being failed by the negotiator.*

The signal is then clipped if higher than 1.0, and while it's 1.0 - the negotiator maliciously tries to fail the network:

1. Negotiator confuses the synapse gates deliberately (line 216):

       route = train_x if intelligence_signal==float(1) else biglatent

*Hopefully this will cause the network to make a wrong prediction.*

2. If a wrong prediction is made while a flag of 1.0 signal is turned on (line 254):

       if intelligence_signal == float(1):
           wake = True
        
*The variable **wake** being set to True causes a sudden reset of all training variables (local, not backend), along with the high gate errors being entered into the recursive cycle. The operator is trained to predict the correct value while being aware of these cycles and the flags, and is the biggest component in every synapse.*

**Neural cryptography implementation**

Instead of transmitting an intelligence flag of 0 to 1 linearly, shuffle it using a key or multiple keys. The network will output different results for each key and pretent to know nothing if the user does not hold any of those keys.

![Neural cryptography](https://cdn.pixabay.com/photo/2016/03/31/17/58/computer-1294045_960_720.png)

--------------------------------------------------------------------------------

**Usage (experimental - indirect neural fitting): python3 run_with_labels.py**

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

**Usage (experimental): python3 load.py**

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

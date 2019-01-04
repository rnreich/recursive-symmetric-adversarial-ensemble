## IntelliForge - Recursive-symmetric adversarial ensemble learning ##
### Indirect high-resolution neural fitting, scalable neural structure operating on raw data ###

An assembly of neural networks training together to optimize under a negotiator that fails them deliberately

![Recursive-symmetric adversarial ensemble learning - Unsafe, autonomous self-aware generic super artificial intelligence](http://i.hmp.me/m/1ca868a0f5f3c1f6d853517d658a8ca5.png)

--------------------------------------------------------------------------------

**Usage**: python3 wakeup_cycle.py

Training an adversarial ensemble lucid dreaming style. Wait for the spikes.

Training data: https://www.kaggle.com/c/microsoft-malware-prediction/data

**Evaluation and prediction**: python3 wakeup_cycle_evaluate.py

Instead of predicting in a normal cycle which is inefficient for this type of network, an initialization sequence is introduced before entering the prediction loop (line 184):

    syn_rand = np.array([np.random.rand(SYN_SIZE)])
    latent_rand = synapses[x][COMP_GATE_IN].predict(syn_rand)

    synapses[x][COMP_PROJ_1].fit(x=latent_rand, y=flags, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
    synapses[x][COMP_PROJ_2].fit(x=syn_rand, y=flags, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)

This sprays the projectors with flags of 0 over perceptions of random data multiple times, convincing the operator to output the correct results.

This script may easily be modified to predict test.csv and to output the results to a submission file.

### Training process ###

The network is forced to learn with high error rates due to the consequences of reaching a clipped 1.0 signal, followed by a wrong prediction. **But this only affects the predictions when the flags are turned on**.

The intelligence signal trasmitted is comprised of a success pattern probability formula (line 263):

    ssr = lc_success_rate if lc_success_rate >= 0.5 else lc_success_rate / 2
    intelligence_signal = beststreak_odds / cycles * ssr
    if intelligence_signal > 1:
        intelligence_signal = float(1)

*The variable **ssr** reduces the signal by half if the success rate is under 0.5 - it's easy to observe that the pretenders use it as trick to avoid being failed by the negotiator.*

The signal is then clipped if higher than 1.0, and while it's 1.0 - the negotiator maliciously tries to fail the network:

1. Negotiator confuses the synapse gates deliberately (line 216):

       route = train_x if intelligence_signal==float(1) else biglatent

*Hopefully this will cause the network to make a wrong prediction.*

2. If a wrong prediction is made while a flag of 1.0 signal is turned on (line 249):

       if intelligence_signal == float(1):
           wake = True
        
*The variable **wake** being set to True causes a sudden reset of all training variables (local, not backend), along with the high gate errors being entered into the recursive cycle.*

*The operator is trained to KNOW the correct value while being aware of these cycles and the flags, and must be the largest component in every synapse.*

### Neural cryptography implementation ###

Instead of transmitting an intelligence flag of 0 to 1 linearly, shuffle it using a key or multiple keys. The network will output different results for each key and pretend to know nothing if the user does not hold any of those keys.

--------------------------------------------------------------------------------

![Neural cryptography](https://cdn.pixabay.com/photo/2016/03/31/17/58/computer-1294045_960_720.png)

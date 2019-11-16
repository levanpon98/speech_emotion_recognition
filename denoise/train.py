from model import *
from data import *

import sys, getopt

# SPEECH ENHANCEMENT NETWORK
SE_LAYERS = 13  # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64  # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = 6  # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM"  # TYPE OF LAYER NORMALIZATION (NM, SBN or None)
SE_LOSS_TYPE = "FL"  # TYPE OF TRAINING LOSS (L1, L2 or FL)

# FEATURE LOSS NETWORK
LOSS_LAYERS = 14  # NUMBER OF INTERNAL LAYERS
LOSS_BASE_CHANNELS = 32  # NUMBER OF FEATURE CHANNELS PER LAYER IN FIRT LAYER
LOSS_BLK_CHANNELS = 5  # NUMBER OF LAYERS BETWEEN CHANNEL NUMBER UPDATES
LOSS_NORM = "SBN"  # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

SET_WEIGHT_EPOCH = 10  # NUMBER OF EPOCHS BEFORE FEATURE LOSS BALANCE
SAVE_EPOCHS = 10  # NUMBER OF EPOCHS BETWEEN MODEL SAVES

log_file = open("logfile.txt", 'w+')

# COMMAND LINE OPTIONS
datafolder = "dataset"
modfolder = "models"
outfolder = "."

with tf.variable_scope(tf.get_variable_scope()):
    input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1, None, 1])
    clean = tf.compat.v1.placeholder(tf.float32, shape=[None, 1, None, 1])

    enhanced = senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)

    loss_weights = tf.compat.v1.placeholder(tf.float32, shape=[SE_LOSS_LAYERS])
    loss_fn = featureloss(clean, enhanced, loss_weights, loss_layers=SE_LOSS_LAYERS, n_layers=LOSS_LAYERS,
                          norm_type=LOSS_NORM,
                          base_channels=LOSS_BASE_CHANNELS, blk_channels=LOSS_BLK_CHANNELS)

trainset, valset = load_full_data_list(datafolder=datafolder)
trainset, valset = load_full_data(trainset, valset)

opt = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_fn[0],
                                                                    var_list=[var for var in tf.trainable_variables() if
                                                                              var.name.startswith("se_")])

graph = tf.Graph()
sess = tf.compat.v1.InteractiveSession(graph=graph)

tf.compat.v1.global_variables_initializer()

Nepochs = 320
saver = tf.compat.v1.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])

loss_train = np.zeros((len(trainset["innames"]), SE_LOSS_LAYERS + 1))
loss_val = np.zeros((len(valset["innames"]), SE_LOSS_LAYERS + 1))
loss_w = np.ones(SE_LOSS_LAYERS)

#####################################################################################

for epoch in range(1, Nepochs + 1):

    print("Epoch no.%d" % epoch)
    # TRAINING EPOCH ################################################################

    ids = np.random.permutation(len(trainset["innames"]))  # RANDOM FILE ORDER

    for id in tqdm(range(0, len(ids)), file=sys.stdout):

        i = ids[id]  # RANDOMIZED ITERATION INDEX
        inputData = trainset["inaudio"][i]  # LOAD DEGRADED INPUT
        outputData = trainset["outaudio"][i]  # LOAD GROUND TRUTH

        # TRAINING ITERATION
        _, loss_vec = sess.run([opt, loss_fn],
                               feed_dict={input: inputData, clean: outputData, loss_weights: loss_w})

        # SAVE ITERATION LOSS
        loss_train[id, 0] = loss_vec[0]
        for j in range(SE_LOSS_LAYERS):
            loss_train[id, j + 1] = loss_vec[j + 1]

    # PRINT EPOCH TRAINING LOSS AVERAGE
    str = "T: %d\t " % (epoch)

    for j in range(SE_LOSS_LAYERS + 1):
        str += ", %10.6e" % (np.mean(loss_train, axis=0)[j])

    log_file.write(str + "\n")
    log_file.flush()

    # SET WEIGHTS AFTER M EPOCHS
    if epoch == SET_WEIGHT_EPOCH:
        loss_w = np.mean(loss_train, axis=0)[1:]

    # SAVE MODEL EVERY N EPOCHS
    if epoch % SAVE_EPOCHS != 0:
        continue

    saver.save(sess, outfolder + "/se_model.ckpt")

    # VALIDATION EPOCH ##############################################################

    print("Validation epoch")

    for id in tqdm(range(0, len(valset["innames"])), file=sys.stdout):

        i = id  # NON-RANDOMIZED ITERATION INDEX
        inputData = valset["inaudio"][i]  # LOAD DEGRADED INPUT
        outputData = valset["outaudio"][i]  # LOAD GROUND TRUTH

        # VALIDATION ITERATION
        output, loss_vec = sess.run([enhanced, loss_fn],
                                    feed_dict={input: inputData, clean: outputData, loss_weights: loss_w})

        # SAVE ITERATION LOSS
        loss_val[id, 0] = loss_vec[0]
        if SE_LOSS_TYPE == "FL":
            for j in range(SE_LOSS_LAYERS):
                loss_val[id, j + 1] = loss_vec[j + 1]

    # PRINT VALIDATION EPOCH LOSS AVERAGE
    str = "V: %d " % (epoch)

    for j in range(SE_LOSS_LAYERS + 1):
        str += ", %10.6e" % (np.mean(loss_val, axis=0)[j] * 1e9)

    log_file.write(str + "\n")
    log_file.flush()

log_file.close()
sess.close()


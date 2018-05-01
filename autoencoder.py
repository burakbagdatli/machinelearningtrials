""" Learning autoencoders with Tensorflow. """
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Training Parameters
LEARNING_RATE = 0.01
NUM_STEPS = 30000
BATCH_SIZE = 256
#
DISPLAY_STEP = 1000
EXAMPLES_TO_SHOW = 10
# Network Parameters
NUM_HIDDEN_1 = 256 # 1st layer num features
NUM_HIDDEN_2 = 128 # 2nd layer num features (the latent dim)
NUM_INPUT = 784 # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
INDEP_VAR = tf.placeholder("float", [None, NUM_INPUT])
WEIGHTS = {'encoder_h1': tf.Variable(tf.random_normal([NUM_INPUT, NUM_HIDDEN_1])),
           'encoder_h2': tf.Variable(tf.random_normal([NUM_HIDDEN_1, NUM_HIDDEN_2])),
           'decoder_h1': tf.Variable(tf.random_normal([NUM_HIDDEN_2, NUM_HIDDEN_1])),
           'decoder_h2': tf.Variable(tf.random_normal([NUM_HIDDEN_1, NUM_INPUT]))}
BIASES = {'encoder_b1': tf.Variable(tf.random_normal([NUM_HIDDEN_1])),
          'encoder_b2': tf.Variable(tf.random_normal([NUM_HIDDEN_2])),
          'decoder_b1': tf.Variable(tf.random_normal([NUM_HIDDEN_1])),
          'decoder_b2': tf.Variable(tf.random_normal([NUM_INPUT]))}
#
def encoder(input_signal):
    """ This builds the encoder """
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_signal, WEIGHTS['encoder_h1']), BIASES['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, WEIGHTS['encoder_h2']), BIASES['encoder_b2']))
    return layer_2
def decoder(encoded):
    """ This builds the decoder """
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoded, WEIGHTS['decoder_h1']), BIASES['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, WEIGHTS['decoder_h2']), BIASES['decoder_b2']))
    return layer_2
# Construct model
ENCODER_OP = encoder(INDEP_VAR)
DECODER_OP = decoder(ENCODER_OP)
# Define loss and optimizer, minimize the squared error
LOSS = tf.reduce_mean(tf.pow(INDEP_VAR - DECODER_OP, 2)) # "auto"encoder: compare output to itself
OPTIMIZER = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(LOSS)
# Initialize the variables (i.e. assign their default value)
INIT = tf.global_variables_initializer()
# Start Training
with tf.Session() as sess:
    sess.run(INIT)
    # Training
    for i in range(1, NUM_STEPS+1):
        # Prepare Data: Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = MNIST.train.next_batch(BATCH_SIZE)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, batch_loss = sess.run([OPTIMIZER, LOSS], feed_dict={INDEP_VAR: batch_x})
        # Display logs per step
        if i % DISPLAY_STEP == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, batch_loss))
    # Testing: Encode and decode images from test set and visualize their reconstruction.
    NUM_IMG = 4
    CANVAS_ORIG = np.empty((28 * NUM_IMG, 28 * NUM_IMG))
    CANVAS_RECON = np.empty((28 * NUM_IMG, 28 * NUM_IMG))
    for i in range(NUM_IMG):
        # MNIST test set
        batch_x, _ = MNIST.test.next_batch(NUM_IMG)
        # Encode and decode the digit image
        g = sess.run(DECODER_OP, feed_dict={INDEP_VAR: batch_x})
        # Display original images
        for j in range(NUM_IMG):
            # Draw the original digits
            CANVAS_ORIG[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(NUM_IMG):
            # Draw the reconstructed digits
            CANVAS_RECON[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])
    print("Original Images")
    plt.figure(figsize=(NUM_IMG, NUM_IMG))
    plt.imshow(CANVAS_ORIG, origin="upper", cmap="gray")
    plt.show()
    print("Reconstructed Images")
    plt.figure(figsize=(NUM_IMG, NUM_IMG))
    plt.imshow(CANVAS_RECON, origin="upper", cmap="gray")
plt.show()

import numpy as np
import tensorflow as tf


def leakey_relu(x):
    return tf.math.maximum(0.2 * x, x)


def l1_loss(y, y_hat):
    return tf.math.reduce_mean(tf.math.abs(y - y_hat))


def l2_loss(y, y_hat):
    return tf.math.reduce_mean(tf.math.square(y - y_hat))


def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0 * x + w1 * tf.nn.batch_normalization(x)


def signal_to_dilated(signal, dilation, n_channels):
    shape = tf.shape(signal)
    pad_elements = dilation - 1 - (shape[2] + dilation - 1) % dilation
    dilated = tf.pad(signal, [[0, 0], [0, 0], [0, pad_elements], [0, 0]])
    dilated = tf.reshape(dilated, [shape[0], -1, dilation, n_channels])
    return tf.transpose(dilated, perm=[0, 2, 1, 3]), pad_elements


def dilated_to_signal(dilated, pad_elements, n_channels):
    shape = tf.shape(dilated)
    signal = tf.transpose(dilated, perm=[0, 2, 1, 3])
    signal = tf.reshape(signal, [shape[0], 1, -1, n_channels])
    return signal[:, :, :shape[1] * shape[2] - pad_elements, :]

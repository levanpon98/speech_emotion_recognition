import tensorflow as tf
import numpy as np
from utils.networks.base_network import Net


class VGG16(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        self.x = tf.placeholder(tf.float32, name='x', shape=[self.config.batch_size,
                                                             self.config.image_width,
                                                             self.config.image_height,
                                                             self.config.image_depth], )
        self.y = tf.placeholder(tf.int16, name='y', shape=[self.config.batch_size,
                                                           self.config.n_classes])
        self.loss = None
        self.accuracy = None
        self.summary = []
        self.regularization = None

    def init_saver(self):
        pass

    def get_summary(self):
        return self.summary

    def conv(self, layer_name, bottom, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1]):
        in_channels = bottom.get_shape()[-1]
        with tf.variable_scope(layer_name):
            w = tf.get_variable(name='weights',
                                trainable=self.config.is_pretrain,
                                shape=[kernel_size[0], kernel_size[1],
                                       in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=self.config.is_pretrain,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            bottom = tf.nn.conv2d(bottom, w, stride, padding='SAME', name='conv')
            bottom = tf.nn.bias_add(bottom, b, name='bias_add')
            bottom = tf.nn.relu(bottom, name='relu')
            return bottom, w, b

    def pool(self, layer_name, bottom, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
        with tf.name_scope(layer_name):
            if is_max_pool:
                bottom = tf.nn.max_pool(bottom, kernel, stride, padding='SAME', name=layer_name)
            else:
                bottom = tf.nn.avg_pool(bottom, kernel, stride, padding='SAME', name=layer_name)
            return bottom

    def fc(self, layer_name, bottom, out_nodes):
        shape = bottom.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name):
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(bottom, [-1, size])
            bottom = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            bottom = tf.nn.relu(bottom)
            return bottom, w, b

    def dropout(self, layer_name, bottom, rate):
        with tf.name_scope(layer_name):
            bottom = tf.layers.dropout(bottom, rate=rate)
            return bottom

    def batch_normalization(self, layer_name, bottom, training=True):
        with tf.name_scope(layer_name):
            epsilon = 1e-3
            bottom = tf.layers.batch_normalization(bottom, epsilon=epsilon, training=training)
            return bottom

    def cal_loss(self, logits, labels, regularization):
        with tf.name_scope('loss') as scope:
            print(regularization)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                    name='cross-entropy') + 0.01 * regularization
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            loss_summary = tf.summary.scalar(scope, self.loss)
            self.summary.append(loss_summary)

    def cal_accuracy(self, logits, labels):
        with tf.name_scope('accuracy') as scope:
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.cast(correct, tf.float32)
            self.accuracy = tf.reduce_mean(correct) * 100.0
            accuracy_summary = tf.summary.scalar(scope, self.accuracy)
            self.summary.append(accuracy_summary)

    def optimize(self):
        with tf.name_scope('optimizer'):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            return train_op

    def build_model(self):

        self.conv1_1, w1, b1 = self.conv('conv1_1', self.x, 64, stride=[1, 1, 1, 1])
        self.pool1 = self.pool('pool1', self.conv1_1, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        self.batch_norm1 = self.batch_normalization('batch_norm1', self.pool1, training=self.is_training)
        self.dropout_1 = self.dropout('dropout_1', self.batch_norm1, rate=0.2)

        self.conv2_1, w2, b2 = self.conv('conv2_1', self.dropout_1, 128, stride=[1, 1, 1, 1])
        self.pool2 = self.pool('pool2', self.conv2_1, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        self.batch_norm2 = self.batch_normalization('batch_norm2', self.pool2, training=self.is_training)
        self.dropout_2 = self.dropout('dropout_2', self.batch_norm2, rate=0.2)

        self.conv3_1, w3, b3 = self.conv('conv3_1', self.dropout_2, 256, stride=[1, 1, 1, 1])
        self.pool3 = self.pool('pool3', self.conv3_1, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        self.batch_norm3 = self.batch_normalization('batch_norm3', self.pool3, training=self.is_training)
        self.dropout_3 = self.dropout('dropout_3', self.batch_norm3, rate=0.2)

        self.fc6, w4, b4 = self.fc('fc6', self.dropout_3, out_nodes=4096)
        self.batch_norm6 = self.batch_normalization('batch_norm1', self.fc6, training=self.is_training)
        self.fc7, w5, b5 = self.fc('fc7', self.batch_norm6, out_nodes=4096)
        self.batch_norm7 = self.batch_normalization('batch_norm2', self.fc7, training=self.is_training)
        self.logits, w6, b6 = self.fc('fc8', self.batch_norm7, out_nodes=self.config.n_classes)

        regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1)
        regularization = regularization + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2)
        regularization = regularization + tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3)
        regularization = regularization + tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4)
        regularization = regularization + tf.nn.l2_loss(w5) + tf.nn.l2_loss(b5)
        regularization = regularization + tf.nn.l2_loss(w6) + tf.nn.l2_loss(b6)

        self.cal_loss(self.logits, self.y, regularization)
        self.cal_accuracy(self.logits, self.y)
        train_op = self.optimize()
        return train_op

    # def load_with_skip(self, data_path, session, skip_layer):
    #     data_dict = np.load(data_path, encoding='latin1').item()  # type: dict
    #     for key in data_dict.keys():
    #         if key not in skip_layer:
    #             with tf.variable_scope(key, reuse=True, auxiliary_name_scope=False):
    #                 # with tf.variable_scope(key, reuse=True):
    #                 for subkey, data in zip(('weights', 'biases'), data_dict[key]):
    #                     session.run(tf.get_variable(subkey).assign(data))

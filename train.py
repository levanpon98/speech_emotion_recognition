import os
from utils.data_loader.data_loader import SpeechCorpus
from utils.config.config import ConfigReader, TrainNetConfig, DataConfig
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.classification.vgg import VGG16
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import time


def train():
    config_reader = ConfigReader('config.yml')
    train_config = TrainNetConfig(config_reader.get_train_config())
    data_config = DataConfig(config_reader.get_train_config())

    train_log_dir = '/content/drive/My Drive/Data/ESR2019/logs/train/'
    val_log_dir = '/content/drive/My Drive/Data/ESR2019/logs/val/'

    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)

    net = VGG16(train_config)
    train_op = net.build_model()
    summaries = net.get_summary()

    print('===INFO====: Load Data')
    data_loader = SpeechCorpus(data_config, is_train=True, is_shuffle=True)
    data, labels = data_loader.load_data()
    data = np.expand_dims(data, axis=3)
    le = LabelEncoder()
    labels_one_hot = to_categorical(le.fit_transform(labels))

    x_train, x_val, y_train, y_val = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
    print("Training data shape: {}".format(x_train.shape))
    print("Training labels shape: {}".format(y_train.shape))
    print("Valid data shape: {}".format(x_val.shape))
    print("Valid labels shape: {}".format(y_val.shape))
    print('===INFO====: End Load Data')

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_loss_over_time = []
    val_loss_over_time = []
    train_acc_over_time = []
    val_acc_over_time = []
    try:

        for epoch in np.arange(train_config.max_step):
            start = time.time()

            average_train_loss = []
            average_train_acc = []
            average_val_loss = []
            average_val_acc = []

            for step in range(x_train.shape[0] // train_config.batch_size):
                batch_train = x_train[step * train_config.batch_size: (step + 1) * train_config.batch_size, :]
                batch_label = y_train[step * train_config.batch_size: (step + 1) * train_config.batch_size]

                _, train_loss, train_acc = sess.run([train_op, net.loss, net.accuracy],
                                                    feed_dict={net.x: batch_train, net.y: batch_label})
                average_train_loss.append(train_loss)
                average_train_acc.append(train_acc)

            for step in range(x_val.shape[0] // train_config.batch_size):
                batch_val = x_val[step * train_config.batch_size: (step + 1) * train_config.batch_size, :]
                batch_label_val = y_val[step * train_config.batch_size: (step + 1) * train_config.batch_size]
                val_loss, val_acc = sess.run([net.loss, net.accuracy],
                                             feed_dict={net.x: batch_val, net.y: batch_label_val})
                average_val_loss.append(val_loss)
                average_val_acc.append(val_acc)

            train_loss_over_time.append(np.mean(average_train_loss))
            val_loss_over_time.append(np.mean(average_val_loss))
            train_acc_over_time.append(np.mean(average_train_acc))
            val_acc_over_time.append(np.mean(average_val_acc))

            if epoch % 500 == 0 or epoch + 1 == train_config.max_step:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
            end = time.time()

            print(
                '===TRAIN===: Step: {}, train loss: {:.2f}, train accuracy: {:.2f}, '
                'val loss {:.2f}, val accuracy {:.2f}, time: {}'.format(
                    epoch,
                    np.mean(average_train_loss),
                    np.mean(average_train_acc),
                    np.mean(average_val_loss),
                    np.mean(average_val_acc),
                    end - start
                ))
    except tf.errors.OutOfRangeError:
        print('===INFO====: Training completed, reaching the maximum number of steps')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    # summarize history for acc
    fig = plt.figure(figsize=(10, 10))
    plt.plot(train_acc_over_time)
    plt.plot(val_acc_over_time)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('accuracy.png', dpi=fig.dpi)

    # summarize history for loss
    fig = plt.figure(figsize=(10, 10))
    plt.plot(train_loss_over_time)
    plt.plot(val_loss_over_time)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('accuracy.png', dpi=fig.dpi)


if __name__ == '__main__':
    train()

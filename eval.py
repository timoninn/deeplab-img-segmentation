import tensorflow as tf
from tensorflow.contrib import slim

import matplotlib.pyplot as plt


def model(inputs):
    return slim.conv2d(inputs, num_outputs=8, kernel_size=1, activation_fn=None)


def _map_function(x):
    feature = {'train/labels': tf.FixedLenFeature([], tf.string),
               'train/prelogits': tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(x, features=feature)

    label = tf.decode_raw(features['train/labels'], tf.uint8)
    prelogit = tf.decode_raw(features['train/prelogits'], tf.float32)

    label = tf.reshape(label, shape=[129, 129])
    prelogit = tf.reshape(prelogit, shape=[129, 129, 256])

    return (label, prelogit)


dataset = tf.data.TFRecordDataset(['dataset/prelogits.tfrecords'])
dataset = dataset.map(map_func=_map_function)
dataset = dataset.repeat(1)
dataset = dataset.shuffle(buffer_size=3)
dataset = dataset.batch(1)

with tf.Graph().as_default():
    iterator = dataset.make_one_shot_iterator()
    label, prelogit_ = iterator.get_next()
    predictions = model(inputs=prelogit_)
    prediction = tf.argmax(predictions, axis=3)

    model_path = tf.train.latest_checkpoint('tmp/model/')
    init_fn = slim.assign_from_checkpoint_fn(model_path,
                                             slim.get_model_variables(),
                                             ignore_missing_vars=True)

    with tf.Session() as sess:
        init_fn(sess)

        while True:
            label1, result = sess.run([label, prediction])

            plt.imshow(label1[0])
            plt.show()

            plt.imshow(result[0])
            plt.show()

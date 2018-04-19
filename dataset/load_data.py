import glob
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import core.preprocess_utills as preprocess


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


deeplab = preprocess.DeepLab()

with tf.Session() as sess:
    feature = {'train/label': tf.FixedLenFeature([], tf.string),
               'train/color': tf.FixedLenFeature([], tf.string)}

    filename_queue = tf.train.string_input_producer(['train.tfrecords'], num_epochs=1)

    reader = tf.TFRecordReader()
    _, serealized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serealized_example, features=feature)

    label = tf.decode_raw(features['train/label'], tf.uint8)
    color = tf.decode_raw(features['train/color'], tf.uint8)

    label = tf.reshape(label, shape=[129, 129])
    color = tf.reshape(color, shape=[513, 513, 3])

    label, color = tf.train.shuffle_batch([label, color],
                                          batch_size=1,
                                          capacity=9,
                                          min_after_dequeue=5)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    writer = tf.python_io.TFRecordWriter('prelogits.tfrecords')

    while True:
        try:
            label_image, color_image = sess.run([label, color])
            prelogits = deeplab.run_decoder(color_image[0])

            print(prelogits.shape)

            feature = {'train/labels': _bytes_feature(tf.compat.as_bytes(label_image.tostring())),
                       'train/prelogits': _bytes_feature(tf.compat.as_bytes(prelogits.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        except tf.errors.OutOfRangeError:
            print('Finish')
            break

    writer.close()

    coord.request_stop()
    coord.join(threads)
    sess.close()

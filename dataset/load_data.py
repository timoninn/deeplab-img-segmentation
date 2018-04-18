import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    color = tf.reshape(color, shape=[129, 129, 3])

    label, color = tf.train.shuffle_batch([label, color],
                                          batch_size=1,
                                          capacity=9,
                                          min_after_dequeue=1)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    a, b = sess.run([label, color])

    print(np.unique(a[0]))

    plt.imshow(a[0])
    plt.show()

    plt.imshow(b[0])
    plt.show()

    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()

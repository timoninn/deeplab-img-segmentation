import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib import slim


def model(inputs):
    return slim.conv2d(inputs, num_outputs=8, kernel_size=1, activation_fn=None)


with tf.Session() as sess:
    feature = {'train/labels': tf.FixedLenFeature([], tf.string),
               'train/prelogits': tf.FixedLenFeature([], tf.string)}

    filename_queue = tf.train.string_input_producer(['dataset/prelogits.tfrecords'],
                                                    num_epochs=500)

    reader = tf.TFRecordReader()
    _, serealized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serealized_example, features=feature)

    label = tf.decode_raw(features['train/labels'], tf.uint8)
    prelogit = tf.decode_raw(features['train/prelogits'], tf.float32)

    label = tf.reshape(label, shape=[129, 129])
    prelogit = tf.reshape(prelogit, shape=[129, 129, 256])

    label, prelogit = tf.train.shuffle_batch([label, prelogit],
                                             batch_size=4,
                                             capacity=9,
                                             min_after_dequeue=5)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # label = tf.one_hot(label, depth=8, axis=-1)
    # logit = model(prelogit)
    #
    # tf.logging.set_verbosity(tf.logging.INFO)
    #
    # label = tf.reshape(label, shape=(-1, 8))
    # logit = tf.reshape(logit, shape=(-1, 8))
    #
    # loss = tf.losses.softmax_cross_entropy(label, logits=logit)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    # total_loss = slim.losses.get_total_loss()
    # train_op = slim.learning.create_train_op(total_loss=total_loss, optimizer=optimizer)
    #
    # final_loss = slim.learning.train(train_op,
    #                                  logdir='tmp/model/',
    #                                  number_of_steps=500,
    #                                  save_summaries_secs=60,
    #                                  log_every_n_steps=20)

    label_image, prelogit_ = sess.run([label, prelogit])
    print(label_image.shape)

    coord.request_stop()
    coord.join(threads)
    sess.close()

plt.imshow(label_image[0])
plt.show()

with tf.Graph().as_default():
    predictions = model(inputs=prelogit_)
    prediction = tf.argmax(predictions, axis=3)

    model_path = tf.train.latest_checkpoint('tmp/model/')
    init_fn = slim.assign_from_checkpoint_fn(model_path,
                                             slim.get_model_variables(),
                                             ignore_missing_vars=True)

    with tf.Session() as sess:
        init_fn(sess)
        result = sess.run(prediction)

    plt.imshow(result[0])
    plt.show()

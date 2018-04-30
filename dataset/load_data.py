import glob
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset.utils as dutils
import core.model as model
import core.preprocess_utils as cutils
import dataset.preprocess_dataset as prd

deeplab = model.DeepLab()

with tf.Session() as sess:
    filename_queue = tf.train.string_input_producer(['../tmp/train.tfrecord'], num_epochs=1)

    reader = tf.TFRecordReader()
    _, serealized_example = reader.read(filename_queue)

    origin_image, seg_image = dutils.parse_tfexample_to_image_seg(serealized_example)

    origin_image, seg_image = tf.train.shuffle_batch([origin_image, seg_image],
                                                     batch_size=1,
                                                     capacity=9,
                                                     min_after_dequeue=5)

    origin_image = cutils.resize_imgs(origin_image,
                                      input_size=513)

    seg_image = cutils.resize_imgs(seg_image,
                                   input_size=129)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    writer = tf.python_io.TFRecordWriter('../tmp/prelogits.tfrecord')

    while True:
        try:
            origin_image_, seg_image_ = sess.run([origin_image, seg_image])

            prd.visualize_segmentation(origin_image_[0], seg_image_[0])

            # dec_output = deeplab.run_decoder(origin_image_[0])
            #
            # print(dec_output[0].shape)
            #
            # example = dutils.decoder_seg_to_tfexample(dec_output[0], seg_image_[0])
            # writer.write(example.SerializeToString())

        except tf.errors.OutOfRangeError:
            print('Finish')
            break

    writer.close()

    coord.request_stop()
    coord.join(threads)
    sess.close()

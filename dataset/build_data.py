import tensorflow as tf
from PIL import Image
import numpy as np


def load_image(path, type):
    """
    Load image at path.

    :param path: Path to image.
    :param type: Either 'JPG' or 'PNG'
    :return: ndarrray of shape [1, height, width, num_channels].
    """
    image = Image.open(path)
    image = np.expand_dims(image, axis=0)

    if type == 'JPG':
        return image
    elif type == 'PNG':
        return np.expand_dims(image, axis=3)
    else:
        raise ValueError('Unsupported image type')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_seg_to_tfexample(origin_image, seg_image):
    origin_image_bytes = tf.compat.as_bytes(origin_image.tostring())
    seg_image_bytes = tf.compat.as_bytes(seg_image.tostring())

    feature = {'image/origin/encoded': _bytes_feature(origin_image_bytes),
               'image/segmentation/encoded': _bytes_feature(seg_image_bytes)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfexample_to_image_seg(example):
    feature = {'image/origin/encoded': tf.FixedLenFeature([], tf.string),
               'image/segmentation/encoded': tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(example, features=feature)

    origin_image = tf.decode_raw(features['image/origin/encoded'], tf.uint8)
    seg_image = tf.decode_raw(features['image/segmentation/encoded'], tf.uint8)

    origin_image = tf.reshape(origin_image, shape=[513, 513, 3])
    seg_image = tf.reshape(seg_image, shape=[129, 129, 1])

    return (origin_image, seg_image)


def decoder_seg_to_tfexample(dec_output, seg_image):
    dec_output_bytes = tf.compat.as_bytes(dec_output.tostring())
    seg_image_bytes = tf.compat.as_bytes(seg_image.tostring())

    feature = {'decoder/output/encoded': _bytes_feature(dec_output_bytes),
               'image/segmentation/encoded': _bytes_feature(seg_image_bytes)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfexample_to_decoder_seg(example):
    feature = {'decoder/output/encoded': tf.FixedLenFeature([], tf.string),
               'image/segmentation/encoded': tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(example, features=feature)

    dec_output = tf.decode_raw(features['decoder/output/encoded'], tf.float32)
    seg_image = tf.decode_raw(features['image/segmentation/encoded'], tf.uint8)

    dec_output = tf.reshape(dec_output, shape=[129, 129, 256])
    seg_image = tf.reshape(seg_image, shape=[129, 129, 1])

    return (dec_output, seg_image)

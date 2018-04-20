import tensorflow as tf


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

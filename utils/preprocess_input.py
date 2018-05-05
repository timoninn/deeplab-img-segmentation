import numpy as np
import tensorflow as tf


def extract_patches(image):
    """
    Extract 9 patches from image. Patched don't overlap. Patch shape [1, 904, 1128, num_channels]

    :param image: Tensor with shape [1, 2710, 3384, num_channels].
    :return: Tensor with shape [9, 904, 1128, num_channels].
    """
    num_channels = tf.shape(image)[3]
    image = tf.extract_image_patches(image,
                                     ksizes=[1, 904, 1128, 1],
                                     strides=[1, 904, 1128, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='SAME')
    return tf.reshape(image, shape=[9, 904, 1128, num_channels])


def map_to_classes(image):
    """
    Map original segmentation map to clasess:
    car, 33
    motorbicycle, 34
    bicycle, 35
    person, 36
    truck, 38
    bus, 39
    tricycle, 40

    :param image: Original segmentation map with shape [batch_size, height, width, 1] or [height, width, 1]
    :return: Mapped segmentation map with same shape as input.
    """
    image = image // 1000
    seg_map = np.zeros(shape=image.shape, dtype=np.uint8)
    seg_map[image == 33] = 1
    seg_map[image == 34] = 2
    seg_map[image == 35] = 3
    seg_map[image == 36] = 4
    seg_map[image == 38] = 5
    seg_map[image == 39] = 6
    seg_map[image == 40] = 7

    return seg_map

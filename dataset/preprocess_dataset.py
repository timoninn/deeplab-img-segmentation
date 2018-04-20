import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from PIL import Image

LABEL_NAMES = np.array(['backgound', 'car', 'motorbicycle', 'bicycle', 'person', 'truck', 'bus', 'tricycle'])


def _crop(image,
          offset_height,
          offset_width,
          target_height=513,
          target_width=513):
    return tf.image.crop_to_bounding_box(image,
                                         offset_height,
                                         offset_width,
                                         target_height=target_height,
                                         target_width=target_width)


def _smart_crop(color_image,
                label_image,
                num_crops=1):
    image_size = tf.shape(label_image)

    image_height = image_size[0]
    image_width = image_size[1]

    offset_height = tf.random_uniform(shape=[num_crops],
                                      minval=0,
                                      maxval=image_height - CROP_SIZE_HEIGHT,
                                      dtype=tf.int32)

    offset_width = tf.random_uniform(shape=[num_crops],
                                     minval=0,
                                     maxval=image_width - CROP_SIZE_WIDTH,
                                     dtype=tf.int32)

    color_image_cropped = crop(color_image,
                               offset_height=offset_height[0],
                               offset_width=offset_width[0])

    label_image_cropped = crop(label_image,
                               offset_height=offset_height[0],
                               offset_width=offset_width[0])

    return (color_image_cropped, label_image_cropped)


def extract_patches(image):
    """
    Extract nine patches from image. Patched don't overlap.

    :param image: Tensor with shape [batch_size, 2710, 3384, num_channels].
    :return:  Tensor with shape [9, 904, 1128, num_channels].
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


def create_clolormap(num_classes):
    """
    Create colormap.

    :param num_classes: Number of classes.
    :return: np.array with shape [num_classes, 3].
    """
    colormap = np.zeros([num_classes, 3], dtype=np.uint8)

    for i in range(num_classes):
        for channel in range(3):
            colormap[i, channel] = (i + channel) * 20

    return colormap


def visualize_segmentation(image, seg_map):
    """
    Visualize segmentation result.

    :param image: Original image with shape [height, width, num_channels].
    :param seg_map: Segmentation map with shape [height, width, 1].
    """

    def _label2color(label):
        colormap = create_clolormap(len(LABEL_NAMES))
        return colormap[label]

    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(image)
    plt.axis('off')

    color_seg_map = _label2color(seg_map)
    plt.subplot(142)
    plt.imshow(color_seg_map)
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(image)
    plt.imshow(color_seg_map, alpha=0.8)
    plt.axis('off')

    unique_labels = np.unique(seg_map)
    unique_colors = _label2color(unique_labels)
    unique_colors = np.expand_dims(unique_colors, 1)

    ax = plt.subplot(144)
    plt.imshow(unique_colors)
    ax.yaxis.tick_right()
    plt.yticks(range(unique_labels.shape[0]), LABEL_NAMES[unique_labels])
    plt.show()


label_image = Image.open('../data/train/label/171206_034513043_Camera_6_instanceIds.png')

label_image = np.array(label_image)
label_image = np.expand_dims(label_image, axis=2)
label_image = map_to_classes(label_image)

color_image = Image.open('../data/train/color/171206_034513043_Camera_6.jpg')

# label_image = np.squeeze(label_image)
# print(np.unique(label_image))
# visualize_segmentation(color_image, label_image)


with tf.Session() as sess:
    color_image = np.expand_dims(color_image, axis=0)
    label_image = np.expand_dims(label_image, axis=0)

    color_image = extract_patches(color_image)
    label_image = extract_patches(label_image)

    color_image, label_image = sess.run([color_image, label_image])

label_image = np.squeeze(label_image)

visualize_segmentation(color_image[4], label_image[4])

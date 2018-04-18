import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

import numpy as np

CROP_SIZE_HEIGHT = 513
CROP_SIZE_WIDTH = 513

color_image_filenames = tf.constant(['../data/train/color/170908_062104003_Camera_6.jpg'])
label_image_filenames = tf.constant(['../data/train/label/170908_062104003_Camera_6_instanceIds.png'])


def crop(image,
         offset_height,
         offset_width,
         target_height=CROP_SIZE_HEIGHT,
         target_width=CROP_SIZE_WIDTH):
    return tf.image.crop_to_bounding_box(image,
                                         offset_height,
                                         offset_width,
                                         target_height=target_height,
                                         target_width=target_width)


def smart_crop(color_image,
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


def _parse_function(color_image_filename,
                    label_image_filename):
    color_image_string = tf.read_file(color_image_filename)
    color_image_decoded = tf.image.decode_image(color_image_string)

    label_image_string = tf.read_file(label_image_filename)
    label_image_decoded = tf.image.decode_image(label_image_string)

    return smart_crop(color_image_decoded, label_image_decoded)


def map_to_classes(image):
    """
    car, 33
    motorbicycle, 34
    bicycle, 35
    person, 36
    truck, 38
    bus, 39
    tricycle, 40
    """

    result = np.zeros(shape=image.shape, dtype=np.uint8)
    result[image == 33] = 1
    result[image == 34] = 2
    result[image == 35] = 3
    result[image == 36] = 4
    result[image == 38] = 5
    result[image == 39] = 6
    result[image == 40] = 7

    return result


def preprocess_raw_data(path):
    filenames = os.listdir(path)

    for filename in filenames:
        if filename == '.DS_Store' or filename == 'mapped':
            continue
        print(filename)
        image = Image.open(path+filename)
        mapped_image = map_to_classes(np.array(image) // 1000)
        Image.fromarray(mapped_image).save(path + 'mapped/' + filename)

def check_images(path):
    filenames = os.listdir(path)

    for filename in filenames:
        if filename == '.DS_Store' or filename == 'mapped':
            continue
        print(filename)
        image = Image.open(path+filename)
        print(np.unique(image))
        plt.imshow(np.squeeze(image))
        plt.show()


def main():
    with tf.Session() as sess:
        dataset = tf.data.Dataset.from_tensor_slices((color_image_filenames, label_image_filenames))
        dataset = dataset.map(_parse_function)

        iterator = dataset.make_one_shot_iterator()
        color_image_cropped, label_image_cropped = iterator.get_next()

        # output_color = tf.image.encode_png(color_image_cropped)
        # output_label = tf.image.encode_png(label_image_cropped)

        color_img, label_img = sess.run([color_image_cropped, label_image_cropped])
        label_img = map_to_classes(label_img)

        Image.fromarray(color_img).save('1.png')
        Image.fromarray(np.squeeze(label_img)).save('2.png')

        # file1 = tf.write_file('1.png', color_img)
        # file2 = tf.write_file('2.png', label_img)

        # sess.run([file1, file2])


main()
# check_label()
# preprocess_raw_data('../data/train_label_batch/')
# check_images('../data/train_label_batch/mapped/')
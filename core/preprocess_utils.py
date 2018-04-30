import tensorflow as tf
from PIL import Image

INPUT_SIZE = 513


def resize_img(image):
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    return resized_image


def resize_imgs(images, input_size):
    shape = tf.shape(images)
    height = shape[1]
    width = shape[2]

    resize_ratio = input_size / tf.maximum(height, width)

    target_size = (resize_ratio * tf.cast(height, tf.float64), resize_ratio * tf.cast(width, tf.float64))
    target_size = tf.cast(target_size, tf.int32)

    return tf.image.resize_images(images,
                                  size=(input_size, input_size),
                                  method=1,
                                  align_corners=True)

    return tf.image.resize_images(images,
                                  size=target_size,
                                  method=1,
                                  align_corners=True)


def pad_to_bounding_box(image):
    qq = tf.cast(image, dtype=tf.float32) - 127.5

    resized_img = tf.image.pad_to_bounding_box(image=qq,
                                               offset_height=0,
                                               offset_width=0,
                                               target_height=INPUT_SIZE,
                                               target_width=INPUT_SIZE)

    rezzz = resized_img + 127.5
    return rezzz


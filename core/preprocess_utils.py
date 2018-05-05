import tensorflow as tf

INPUT_SIZE = 513


def resize_images(images, size, save_ratio=True):
    shape = tf.shape(images)
    height = tf.to_float(shape[1])
    width = tf.to_float(shape[2])

    resize_ratio = size / tf.maximum(height, width)

    target_size = (resize_ratio * height, resize_ratio * width)
    target_size = tf.to_int32(target_size)

    images = tf.image.resize_bilinear(images,
                                      size=target_size if save_ratio == True else (size, size),
                                      align_corners=True)

    return tf.cast(images, tf.uint8)


def pad_to_bounding_box(image):
    MEAN = 127.5

    image = tf.to_float(image) - MEAN
    image = tf.image.pad_to_bounding_box(image=image,
                                         offset_height=0,
                                         offset_width=0,
                                         target_height=INPUT_SIZE,
                                         target_width=INPUT_SIZE)
    return image + MEAN


def _crop(image,
          offset_height,
          offset_width,
          target_height=513,
          target_width=513):
    """
    Deprecated. Will be removed on release.
    """
    return tf.image.crop_to_bounding_box(image,
                                         offset_height,
                                         offset_width,
                                         target_height=target_height,
                                         target_width=target_width)

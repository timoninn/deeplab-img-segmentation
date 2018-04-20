import tensorflow as tf
from PIL import Image

INPUT_SIZE = 513


def resize_img(image):
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    return resized_image


def pad_to_bounding_box(image):
    img_placeholder = tf.placeholder(shape=(None, None, 3), dtype=tf.float32)

    rrr = tf.constant(shape=[1], dtype=tf.float32, value=127.5)
    qq = img_placeholder - rrr

    resized_img = tf.image.pad_to_bounding_box(image=qq,
                                               offset_height=0,
                                               offset_width=0,
                                               target_height=INPUT_SIZE,
                                               target_width=INPUT_SIZE)

    rezzz = resized_img + rrr
    with tf.Session() as sess:
        return sess.run(rezzz, feed_dict={img_placeholder: image})

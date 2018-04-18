import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np

label_path = '../data/train_label_batch/*.png'
color_path = '../data/train_color_batch/*.jpg'

labels = glob.glob(label_path)
colors = glob.glob(color_path)

labels = sorted(labels)
colors = sorted(colors)

assert len(colors) == len(labels), 'Number of examples should match'

num_examples = len(colors)
print(num_examples)

train_labels = labels[0:int(0.6 * num_examples)]
train_colors = colors[0:int(0.6 * num_examples)]

val_labels = labels[int(0.6 * num_examples):int(0.8 * num_examples)]
val_colors = colors[int(0.6 * num_examples):int(0.8 * num_examples)]

test_labels = labels[int(0.8 * num_examples):]
test_colors = colors[int(0.8 * num_examples):]


def load_image(path):
    image = Image.open(path)
    return np.array(image)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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
    image = image // 1000

    result = np.zeros(shape=image.shape, dtype=np.uint8)
    result[image == 33] = 1
    result[image == 34] = 2
    result[image == 35] = 3
    result[image == 36] = 4
    result[image == 38] = 5
    result[image == 39] = 6
    result[image == 40] = 7

    return result


def resize_image(image,
                 target_height,
                 target_width):
    with tf.Session() as sess:
        resize_image = tf.image.resize_images(image, size=[target_height, target_width], method=1)
        resize_image = tf.cast(resize_image, dtype=tf.uint8)
        resized_image = sess.run(resize_image)

        return resized_image


writer = tf.python_io.TFRecordWriter('train.tfrecords')

for i in range(len(train_labels)):
    label_image = load_image(train_labels[i])
    color_image = load_image(train_colors[i])

    label_image = map_to_classes(label_image)
    label_image = np.expand_dims(label_image, axis=2)
    label_image = resize_image(label_image,
                               target_height=129,
                               target_width=129)

    color_image = resize_image(color_image,
                               target_height=513,
                               target_width=513)

    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label_image.tostring())),
               'train/color': _bytes_feature(tf.compat.as_bytes(color_image.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()

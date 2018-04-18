import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from core import deeplab

from PIL import Image
import matplotlib.pyplot as plt

model_dir_path = '../frozen_inference_graph/deeplabv3_pascal_train_aug/'

def resize_img(image):
    INPUT_SIZE = 513

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
                                               target_height=513,
                                               target_width=513)

    rezzz = resized_img + rrr
    with tf.Session() as sess:
        return sess.run(rezzz, feed_dict={img_placeholder: image})


class DeepLab(object):
    INPUT_TENSOR_NAME = 'add_2:0'
    SEMATIC_PREDICITON_TENSOR_NAME = 'logits/semantic/BiasAdd:0'
    DECODER_OUTPUT_TENSOR_NAME = 'decoder/decoder_conv1_pointwise/Relu:0'

    def __init__(self):
        self.graph = tf.Graph()

        file = open(model_dir_path + 'frozen_inference_graph.pb', mode='rb')
        graph_def = tf.GraphDef.FromString(file.read())
        file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run_decoder(self, inputs):
        return self.sess.run(self.DECODER_OUTPUT_TENSOR_NAME,
                             feed_dict={self.INPUT_TENSOR_NAME: inputs})

    def run_semantic(self, inputs):
        return self.sess.run(self.SEMATIC_PREDICITON_TENSOR_NAME,
                             feed_dict={self.INPUT_TENSOR_NAME: inputs})


# image = Image.open('../data/train/color/171206_034456206_Camera_6.jpg')
# image = Image.open('../data/train/0.jpg')
# image = resize_img(image)
# image = pad_to_bounding_box(image)
#
# deeplab = DeepLab()
# result = deeplab.run_semantic(image)
# with tf.Session() as sess:
#     upsampled_logits = tf.image.resize_bilinear(images=result, size=(513, 513), align_corners=True)
#     prediction = tf.argmax(upsampled_logits, axis=3)
#
#     result = sess.run(prediction)
#
# plt.imshow(np.squeeze(result))
# plt.show()
# print(result.shape)

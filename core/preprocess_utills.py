import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from core import deeplab

ckpt_dir = '/Users/nikki/Desktop/models/research/deeplab/graph/1/'
model_dir_path = '/Users/nikki/Development/deeplab-img-segmentation/model_graph/deeplabv3_pascal_train_aug/'

INPUT_TENSOR_NAME = 'xception_65/Pad:0'
OUTPUT_TENSOR_NAME = 'xception_65/entry_flow/block3/unit_1/xception_module/add' + ':0'


zeros = np.zeros(shape=(1, 515, 515, 3), dtype=np.float32)
ones = np.ones(shape=(1, 515, 515, 3), dtype=np.float32)

with tf.Graph().as_default() as graph:
    with slim.arg_scope([slim.batch_norm],
                        is_training=False,
                        scale=True):

        inputs = tf.placeholder(tf.float32, shape=(1, 515, 515, 3))
        model = deeplab.xception(inputs,
                                   scope='xception_65')

        with tf.Session() as sess:
            variables_to_restore = slim.get_variables_to_restore()
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess,
                             save_path=model_dir_path + 'model.ckpt')

            outputs = sess.run(OUTPUT_TENSOR_NAME,
                               feed_dict={inputs: ones})

    writer = tf.summary.FileWriter(logdir=ckpt_dir+'3', graph=graph)




class DeepLab(object):
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


    def run(self, inputs):
        return self.sess.run(OUTPUT_TENSOR_NAME,
                             feed_dict={INPUT_TENSOR_NAME: inputs})



deeplab = DeepLab()
groundthruth_outputs = deeplab.run(ones)


with tf.Session() as sess:
    if sess.run(tf.reduce_all(tf.equal(groundthruth_outputs, outputs))):
        print('Equal')
    else:
        print('Not equal')
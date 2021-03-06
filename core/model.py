import tensorflow as tf
from tensorflow.contrib import slim

FROSEN_INFERENCE_GRAPH_DIR_PATH = '../frozen_inference_graph/deeplabv3_pascal_train_aug/'


class DeepLab(object):
    INPUT_TENSOR_NAME = 'add_2:0'
    SEMATIC_PREDICITON_TENSOR_NAME = 'logits/semantic/BiasAdd:0'
    DECODER_OUTPUT_TENSOR_NAME = 'decoder/decoder_conv1_pointwise/Relu:0'

    def __init__(self, path=None):
        self.graph = tf.Graph()

        path = FROSEN_INFERENCE_GRAPH_DIR_PATH + 'frozen_inference_graph.pb' if path==None else path

        file = open(path, mode='rb')
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


class Logits(object):
    def layer(self, inputs):
        return slim.conv2d(inputs,
                           num_outputs=8,
                           kernel_size=1,
                           activation_fn=None,
                           weights_regularizer=slim.l2_regularizer(1e-4))

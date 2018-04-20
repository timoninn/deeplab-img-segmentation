import tensorflow as tf

FROSEN_INFERENCE_GRAPH_DIR_PATH = '../frozen_inference_graph/deeplabv3_pascal_train_aug/'

class DeepLab(object):
    INPUT_TENSOR_NAME = 'add_2:0'
    SEMATIC_PREDICITON_TENSOR_NAME = 'logits/semantic/BiasAdd:0'
    DECODER_OUTPUT_TENSOR_NAME = 'decoder/decoder_conv1_pointwise/Relu:0'

    def __init__(self):
        self.graph = tf.Graph()

        file = open(FROSEN_INFERENCE_GRAPH_DIR_PATH + 'frozen_inference_graph.pb', mode='rb')
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

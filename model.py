import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np

from PIL import Image
from matplotlib import gridspec
from matplotlib import pyplot as plt

log_dir = '/Users/nikki/Desktop/models/research/deeplab/graph/1/'
model_dir_path = 'model_graph/deeplabv3_pascal_train_aug/'

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

class DeepLab(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self):
        self.graph = tf.Graph()

        file = open(model_dir_path + 'frozen_inference_graph.pb', mode='rb')
        graph_def = tf.GraphDef.FromString(file.read())
        file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph.')

        graph_def = tf.graph_util.extract_sub_graph(graph_def,
                                                    dest_nodes=['xception_65/Pad', 'ResizeBilinear_2', 'ImageTensor', 'SemanticPredictions'])

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)


    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def save_graph(self):
        writer = tf.summary.FileWriter(logdir=log_dir+'my_net', graph=self.graph)


def visualise_segmentation(image, seg_map)

image = Image.open('images/1.jpg')
deeplab = DeepLab()
resized_img, seg_map = deeplab.run(image)


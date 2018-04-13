import tensorflow as tf
from tensorflow.contrib import slim
from core import deeplab

ckpt_dir = '/Users/nikki/Desktop/models/research/deeplab/graph/1/'

with tf.Graph().as_default() as g:
    inputs = tf.placeholder(tf.float32, shape=(None, 257, 257, 64))

    with slim.arg_scope([slim.batch_norm],
                        is_training=False,
                        scale=True):
        outputs = deeplab.xception(inputs,
                                   scope='xception_65')

    for v in slim.get_model_variables():
        print('name={}, shape={}'.format(v.name, v.get_shape()))

    writer = tf.summary.FileWriter(logdir=ckpt_dir+'3', graph=g)


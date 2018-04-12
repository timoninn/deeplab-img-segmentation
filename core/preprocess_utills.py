import tensorflow as tf
from tensorflow.contrib import slim
from core import deeplab

with tf.Graph().as_default():
    inputs = tf.placeholder(tf.float32, shape=(None, 257, 257, 64))
    outputs = deeplab.xception_module(inputs,
                                      depth_list=[128, 128, 128],
                                      skip_connection_type='conv',
                                      stride=1,
                                      rate=1,
                                      separable_conv_activaation_fn=False,
                                      scope=None)

    for v in slim.get_model_variables():
        print('name={}, shape={}'.format(v.name, v.get_shape()))


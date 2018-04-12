import tensorflow as tf
from tensorflow.contrib import slim


# def xception():
#
# def flow():
#
# def xception_block():
#
# def unit():
#


def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    rate=1,
                    separable_conv_activaation_fn=False,
                    scope=None):
    if len(depth_list) != 3:
        raise ValueError('Expect three elements in depth_list')

    def _separable_conv2d(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate,
                          scope):
        if separable_conv_activaation_fn:
            activation_fn = tf.nn.relu
        else:
            inputs = tf.nn.relu(inputs)
            activation_fn = None

        return separable_convolution(inputs,
                                     num_outputs,
                                     kernel_size,
                                     depth_multiplier=depth_multiplier,
                                     stride=stride,
                                     rate=rate,
                                     activation_fn=activation_fn,
                                     scope=scope)

    residual = inputs
    with tf.variable_scope(scope, 'xception_module', [inputs]):
        for i in range(3):
            residual = _separable_conv2d(residual,
                                         num_outputs=depth_list[i],
                                         kernel_size=3,
                                         depth_multiplier=1,
                                         stride=stride,
                                         rate=rate,
                                         scope='separable_conv' + str(i+1))

    if skip_connection_type == 'conv':
        inputs = slim.conv2d(inputs,
                            num_outputs=depth_list[-1],
                            kernel_size=[1, 1],
                            stride=stride,
                            activation_fn=None,
                            scope='shortcut')
        return residual + inputs
    elif skip_connection_type == 'add':
        return residual + inputs
    elif skip_connection_type == 'none':
        return residual
    else:
        raise ValueError('Unsupported skip connection type.')





def separable_convolution(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate,
                          activation_fn=None,
                          scope=None):
    def _split_separable_conv2d(padding):
        """Split separable conv2d into deptwise and pointwise"""
        outputs = slim.separable_conv2d(inputs,
                                        num_outputs=None,
                                        kernel_size=kernel_size,
                                        depth_multiplier=depth_multiplier,
                                        stride=stride,
                                        padding=padding,
                                        rate=rate,
                                        activation_fn=activation_fn,
                                        scope=scope + '_depthwise')

        return slim.conv2d(outputs,
                           num_outputs,
                           1,
                           activation_fn=activation_fn,
                           scope=scope + '_pointwise')

    return _split_separable_conv2d(padding='SAME')


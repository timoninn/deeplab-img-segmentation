import tensorflow as tf
from tensorflow.contrib import slim


def xception(inputs,
             scope):
    with tf.variable_scope(scope, 'xception', [inputs]):
        with slim.arg_scope([slim.separable_conv2d,
                             slim.conv2d],
                            normalizer_fn=slim.batch_norm):
            inputs = slim.conv2d(inputs,
                                 num_outputs=32,
                                 kernel_size=3,
                                 stride=2,
                                 padding='VALID',
                                 scope='entry_flow/conv1_1')

            inputs = slim.conv2d(inputs,
                                 num_outputs=64,
                                 kernel_size=3,
                                 stride=1,
                                 padding='SAME',
                                 scope='entry_flow/conv1_2')

            inputs = xception_block(inputs,
                                    depth_list=[128, 128, 128],
                                    skip_connetcion_type='conv',
                                    stride=2,
                                    separable_conv_activation_fn=False,
                                    num_units=1,
                                    scope='entry_flow/block1')

            inputs = xception_block(inputs,
                                    depth_list=[256, 256, 256],
                                    skip_connetcion_type='conv',
                                    stride=2,
                                    separable_conv_activation_fn=False,
                                    num_units=1,
                                    scope='entry_flow/block2')

            inputs = xception_block(inputs,
                                    depth_list=[728, 728, 728],
                                    skip_connetcion_type='conv',
                                    stride=1,
                                    separable_conv_activation_fn=False,
                                    num_units=1,
                                    scope='entry_flow/block3')

            inputs = xception_block(inputs,
                                    depth_list=[728, 728, 728],
                                    skip_connetcion_type='add',
                                    stride=1,
                                    separable_conv_activation_fn=False,
                                    num_units=16,
                                    scope='middle_flow/block1')

            inputs = xception_block(inputs,
                                    depth_list=[728, 1024, 1024],
                                    skip_connetcion_type='conv',
                                    stride=1,
                                    separable_conv_activation_fn=False,
                                    num_units=1,
                                    scope='exit_flow/block1')

            inputs = xception_block(inputs,
                                    depth_list=[1536, 1536, 2048],
                                    skip_connetcion_type='none',
                                    stride=1,
                                    separable_conv_activation_fn=False,
                                    num_units=1,
                                    scope='exit_flow/block2')

    return inputs


def xception_block(inputs,
                   depth_list,
                   skip_connetcion_type,
                   stride,
                   separable_conv_activation_fn,
                   num_units,
                   scope):
    with tf.variable_scope(scope, 'block', [inputs]):
        for i in range(num_units):
            inputs = xception_module(inputs,
                                     depth_list,
                                     skip_connetcion_type,
                                     stride=stride,
                                     separable_conv_activation_fn=separable_conv_activation_fn,
                                     scope='unit_' + str(i + 1) + '/xception_module')
    return inputs


def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    rate=1,
                    separable_conv_activation_fn=False,
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
        if separable_conv_activation_fn:
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
                                         stride=stride if i == 2 else 1,
                                         rate=rate,
                                         scope='separable_conv' + str(i + 1))

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
                          use_explict_padding=True,
                          activation_fn=None,
                          scope=None):
    def _separable_conv2d(padding):
        return slim.separable_conv2d(inputs,
                                     num_outputs,
                                     kernel_size,
                                     depth_multiplier=depth_multiplier,
                                     stride=stride,
                                     rate=rate,
                                     activation_fn=activation_fn,
                                     scope=scope)

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

    if stride == 1 or not use_explict_padding:
        return _split_separable_conv2d(padding='SAME')
    else:
        inputs = fixed_padding(inputs, kernel_size, rate)
        return _split_separable_conv2d(padding='VALID')


def fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      rate: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs

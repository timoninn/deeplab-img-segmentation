import tensorflow as tf
from tensorflow.contrib import slim
from core import deeplab

ckpt_dir = '/Users/nikki/Desktop/models/research/deeplab/graph/1/'
model_dir_path = '/Users/nikki/Development/deeplab-img-segmentation/model_graph/deeplabv3_pascal_train_aug/model.ckpt'

with tf.Graph().as_default() as g:
    inputs = tf.placeholder(tf.float32, shape=(None, 513, 513, 3))

    with slim.arg_scope([slim.batch_norm],
                        is_training=False,
                        scale=True):

        outputs = deeplab.xception(inputs,
                                   scope='xception_65')

        variables_to_restore = slim.get_variables_to_restore()
        restorer = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            restorer.restore(sess,
                             save_path=model_dir_path)

        print(outputs.get_shape())

    writer = tf.summary.FileWriter(logdir=ckpt_dir+'3', graph=g)







# def load_model(path):

    # for v in slim.get_model_variables():
    #     print('name={}, shape={}'.format(v.name, v.get_shape()))

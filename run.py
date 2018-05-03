from core import model
import tensorflow as tf
from tensorflow.contrib import slim

import dataset.utils as dutils
import core.preprocess_utils as cutils
import dataset.preprocess_dataset as prdata

deeplab = model.DeepLab(path='frozen_inference_graph/deeplabv3_pascal_train_aug/frozen_inference_graph.pb')
logits = model.Logits()

origin_image_path = 'data/train/color/170908_082012038_Camera_5.jpg'
seg_image_path = 'data/train/label/170908_082012038_Camera_5_instanceIds.png'

with tf.Session() as sess:
    origin_image = dutils.load_image(origin_image_path, 'JPG')
    seg_image = dutils.load_image(seg_image_path, 'PNG')

    seg_image = prdata.map_to_classes(seg_image)

    origin_image_patches = prdata.extract_patches(origin_image)
    segm_image_patches = prdata.extract_patches(seg_image)

    origin_image_patches = cutils.resize_imgs(origin_image_patches, input_size=513)
    segm_image_patches = cutils.resize_imgs(segm_image_patches, input_size=513)

    origin_image_patches, segm_image_patches = sess.run([origin_image_patches, segm_image_patches])

    prelogit_placeholder = tf.placeholder(tf.float32, shape=(1, 129, 129, 256))
    prediction = logits.run(prelogit_placeholder)
    prediction = tf.image.resize_bilinear(prediction,
                                          size=(513, 513),
                                          align_corners=True)
    prediction = tf.argmax(prediction, axis=3)

    model_path = tf.train.latest_checkpoint('tmp/model/')
    init_fn = slim.assign_from_checkpoint_fn(model_path,
                                             slim.get_model_variables(),
                                             ignore_missing_vars=True)
    init_fn(sess)

    for j in range(5, 9):
        prelogit = deeplab.run_decoder(origin_image_patches[j])
        prediction_res = sess.run(prediction, feed_dict={prelogit_placeholder: prelogit})
        prdata.visualize_segmentation(origin_image_patches[j], prediction_res[0])
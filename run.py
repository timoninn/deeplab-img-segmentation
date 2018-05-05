import tensorflow as tf
from tensorflow.contrib import slim

from core import model
from core import preprocess_utils
from dataset import build_data
from utils import preprocess_input
from utils import visualization

deeplab = model.DeepLab(path='frozen_inference_graph/deeplabv3_pascal_train_aug/frozen_inference_graph.pb')
logits = model.Logits()

image_name = '171206_034510697_Camera_5'

origin_image_path = 'data/main/origin/{}.jpg'.format(image_name)
seg_image_path = 'data/main/seg/{}_instanceIds.png'.format(image_name)

with tf.Session() as sess:
    origin_image = build_data.load_image(origin_image_path, 'JPG')
    seg_image = build_data.load_image(seg_image_path, 'PNG')

    seg_image = preprocess_input.map_to_classes(seg_image)

    origin_image_patches, segm_image_patches = preprocess_input.preprocess_input(origin_image,
                                                                                 seg_image,
                                                                                 origin_size=513,
                                                                                 seg_size=513)

    origin_image_patches, segm_image_patches = sess.run([origin_image_patches, segm_image_patches])

    prelogit_placeholder = tf.placeholder(tf.float32, shape=(1, 129, 129, 256))
    prediction = logits.layer(prelogit_placeholder)
    prediction = tf.image.resize_bilinear(prediction,
                                          size=(513, 513),
                                          align_corners=True)
    prediction = tf.argmax(prediction, axis=3)

    model_path = tf.train.latest_checkpoint('tmp/train_log_dir/')
    init_fn = slim.assign_from_checkpoint_fn(model_path,
                                             slim.get_model_variables(),
                                             ignore_missing_vars=True)
    init_fn(sess)

    for j in range(origin_image_patches.shape[0]):
        prelogit = deeplab.run_decoder(origin_image_patches[j])
        prediction_res = sess.run(prediction, feed_dict={prelogit_placeholder: prelogit})

        visualization.visualize_segmentation(origin_image_patches[j], prediction_res[0])

        # visualization.visualize_segmentation(origin_image_patches[j], segm_image_patches[j])

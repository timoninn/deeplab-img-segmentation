import tensorflow as tf
from tensorflow.contrib import slim

import dataset.utils as dutils
from core import model

dataset = tf.data.TFRecordDataset(['tmp/prelogits.tfrecord'])
dataset = dataset.map(map_func=dutils.parse_tfexample_to_decoder_seg)
dataset = dataset.repeat(1)
dataset = dataset.batch(9)

logits = model.Logits()

with tf.Graph().as_default():
    iterator = dataset.make_one_shot_iterator()
    dec_output, seg_image = iterator.get_next()
    predictions = logits.run(dec_output)
    predictions = tf.argmax(predictions, axis=3)

    seg_image = tf.reshape(seg_image, shape=[-1])
    predictions = tf.reshape(predictions, shape=[-1])

    # Define evaluation metric.
    metric_map = {}
    metric_map['miou'] = tf.metrics.mean_iou(seg_image,
                                             predictions=predictions,
                                             num_classes=8)

    metrics_to_values, metrics_to_updates = tf.contrib.metrics.aggregate_metric_map(metric_map)

    for metric_name, metric_value in metrics_to_values.items():
        slim.summaries.add_scalar_summary(metric_value,
                                          name=metric_name,
                                          print_summary=True)

    tf.logging.set_verbosity(tf.logging.INFO)
    slim.evaluation.evaluation_loop(master='',
                                    checkpoint_dir='tmp/model/',
                                    logdir='tmp/eval_log_dir/',
                                    num_evals=5,
                                    eval_op=list(metrics_to_updates.values()),
                                    eval_interval_secs=1)

    # model_path = tf.train.latest_checkpoint('tmp/model/')
    # init_fn = slim.assign_from_checkpoint_fn(model_path,
    #                                          slim.get_model_variables(),
    #                                          ignore_missing_vars=True)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #
    #     init_fn(sess)
    #
    #     while True:
    #         sess.run(list(metrics_to_updates.values()))
    #
    #         metric_values = sess.run(list(metrics_to_values.values()))
    #         print(metric_values)

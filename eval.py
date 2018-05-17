# import sys
# import os.path
#
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from tensorflow.contrib import slim

from core import model
from dataset import build_data
from utils import preprocess_input

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('prelogits_path', None,
                    'Path to .tfrecord prelogits')

flags.DEFINE_string('eval_logdir', None,
                    'Evaluation log directory')

flags.DEFINE_string('checkpoint_dir', None,
                    'Train checkpoint directory')


def _get_iterator(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(map_func=build_data.parse_tfexample_to_decoder_seg)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(9)

    return dataset.make_one_shot_iterator()


logits = model.Logits()


def _eval(iterator, checkpoint_dir, log_dir):
    dec_output, seg_image = iterator.get_next()

    predictions = logits.layer(dec_output)
    predictions = tf.argmax(predictions, axis=3)

    seg_image = tf.to_int64(seg_image)
    seg_image = tf.squeeze(seg_image, axis=3)

    # bicycle, tricycle
    hide_classes = [3, 7]
    predictions = preprocess_input.filter_classes(predictions, class_indices=hide_classes)
    seg_image = preprocess_input.filter_classes(seg_image, class_indices=hide_classes)

    predictions = tf.reshape(predictions, shape=[-1])
    seg_image = tf.reshape(seg_image, shape=[-1])

    # Ignore background class(0).
    weights = tf.not_equal(seg_image, 0)

    # Define evaluation metric.
    metric_map = {}
    metric_map['miou'] = tf.metrics.mean_iou(seg_image,
                                             predictions=predictions,
                                             num_classes=8,
                                             weights=None)

    metrics_to_values, metrics_to_updates = tf.contrib.metrics.aggregate_metric_map(metric_map)

    for metric_name, metric_value in metrics_to_values.items():
        slim.summaries.add_scalar_summary(metric_value,
                                          name=metric_name,
                                          print_summary=True)

    tf.logging.set_verbosity(tf.logging.INFO)
    slim.evaluation.evaluation_loop(master='',
                                    checkpoint_dir=checkpoint_dir,
                                    logdir=log_dir,
                                    num_evals=200,
                                    eval_op=list(metrics_to_updates.values()),
                                    eval_interval_secs=1)


def main(unused_argv):
    iterator = _get_iterator([FLAGS.prelogits_path])
    _eval(iterator,
          checkpoint_dir=FLAGS.checkpoint_dir,
          log_dir=FLAGS.eval_logdir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['prelogits_path', 'eval_logdir', 'checkpoint_dir'])
    tf.app.run()

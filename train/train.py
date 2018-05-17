import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from tensorflow.contrib import slim

from dataset import build_data
from core import model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('prelogits_path', None,
                    'Path to .tfrecord prelogits')

flags.DEFINE_string('train_logdir', None,
                    'Train log directory')

flags.DEFINE_integer('num_steps', None,
                     'Number of train steps')


def _get_iterator(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(build_data.parse_tfexample_to_decoder_seg)
    dataset = dataset.repeat(1000)
    dataset = dataset.prefetch(24)
    dataset = dataset.shuffle(24)
    dataset = dataset.batch(6)

    return dataset.make_one_shot_iterator()


def train(iterator, logdir, num_steps):
    logits = model.Logits()

    prelogits, seg_images = iterator.get_next()

    seg_images = tf.one_hot(seg_images, depth=8, axis=-1)
    logits = logits.layer(prelogits)

    tf.logging.set_verbosity(tf.logging.INFO)

    seg_images = tf.reshape(seg_images, shape=(-1, 8))
    logits = tf.reshape(logits, shape=(-1, 8))

    loss = tf.losses.softmax_cross_entropy(seg_images, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    train_op = slim.learning.create_train_op(total_loss=total_loss, optimizer=optimizer)

    slim.learning.train(train_op,
                        logdir=logdir,
                        number_of_steps=num_steps,
                        save_summaries_secs=2,
                        log_every_n_steps=20,
                        save_interval_secs=30)


def main(unused_argv):
    iterator = _get_iterator([FLAGS.prelogits_path])
    train(iterator,
          logdir=FLAGS.train_logdir,
          num_steps=FLAGS.num_steps)


if __name__ == '__main__':
    flags.mark_flags_as_required(['prelogits_path', 'train_logdir', 'num_steps'])
    tf.app.run()

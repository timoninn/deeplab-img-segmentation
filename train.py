import tensorflow as tf
from tensorflow.contrib import slim
import dataset.utils as dutils
from core import model

model = model.Logits()

dataset = tf.data.TFRecordDataset(['tmp/prelogits.tfrecord'])
dataset = dataset.map(dutils.parse_tfexample_to_decoder_seg)
dataset = dataset.repeat(100)
dataset = dataset.prefetch(18)
dataset = dataset.batch(6)

iterator = dataset.make_one_shot_iterator()
prelogits, seg_images = iterator.get_next()

seg_images = tf.one_hot(seg_images, depth=8, axis=-1)
logits = model.run(prelogits)

tf.logging.set_verbosity(tf.logging.INFO)

seg_images = tf.reshape(seg_images, shape=(-1, 8))
logits = tf.reshape(logits, shape=(-1, 8))

loss = tf.losses.softmax_cross_entropy(seg_images, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
total_loss = slim.losses.get_total_loss()
tf.summary.scalar('losses/total_loss', total_loss)

train_op = slim.learning.create_train_op(total_loss=total_loss, optimizer=optimizer)

final_loss = slim.learning.train(train_op,
                                 logdir='tmp/model/',
                                 number_of_steps=4000,
                                 save_summaries_secs=2,
                                 log_every_n_steps=20)

import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf

from core import model
from dataset import build_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', None,
                    'Path to .tfrecord datafile')

flags.DEFINE_string('output_file', None,
                    'Output filepath')

def _get_iterator(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(build_data.parse_tfexample_to_image_seg)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(2)

    return dataset.make_one_shot_iterator()


def _calculate_dec_outputs(iterator, path):
    deeplab = model.DeepLab()

    origin_image, seg_image = iterator.get_next()

    writer = tf.python_io.TFRecordWriter(path)
    with tf.Session() as sess:
        while True:
            try:
                origin_image_res, seg_image_res = sess.run([origin_image, seg_image])

                dec_output = deeplab.run_decoder(origin_image_res[0])
                print(dec_output.shape)
                print(seg_image_res.shape)

                example = build_data.decoder_seg_to_tfexample(dec_output[0], seg_image_res[0])
                writer.write(example.SerializeToString())

            except tf.errors.OutOfRangeError:
                print('Finish')
                break
    writer.close()


def main(unused_argv):
    iterator = _get_iterator([FLAGS.data_path])
    _calculate_dec_outputs(iterator, FLAGS.output_file)


if __name__ == '__main__':
    flags.mark_flags_as_required(['data_path', 'output_file'])
    tf.app.run()

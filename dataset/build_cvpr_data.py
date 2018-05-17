import tensorflow as tf
import glob

import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import visualization
from utils import preprocess_input

from dataset import build_data

IMAGE_HEIGHT = 2710
IMAGE_WIDTH = 3384

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('images_folder', None,
                    'Folder containing images.')

flags.DEFINE_enum('split_part', None, ['train', 'val', 'test'],
                  'Which part of dataset convert')

flags.DEFINE_multi_float('splits', [0.6, 0.2, 0.2],
                         'Dataset splits')

flags.DEFINE_string('output_file', None,
                    'Output filepath')


def _get_files(path):
    origin_images_path = path + 'origin/*.jpg'
    seg_images_path = path + 'seg/*.png'

    origin_images = glob.glob(origin_images_path)
    seg_images = glob.glob(seg_images_path)

    origin_images = sorted(origin_images)
    seg_images = sorted(seg_images)

    assert len(origin_images) == len(seg_images), 'Number of examples should match'

    return list(zip(origin_images, seg_images))


def _split_dataset(samples, parts):
    num_samples = len(samples)
    num_train = round(num_samples * parts[0])
    num_val = round(num_samples * parts[1])
    num_test = round(num_samples * parts[2])

    train_samples = samples[0:num_train]
    val_samples = samples[num_train:num_train + num_val]
    test_samples = samples[num_train + num_val:num_train + num_val + num_test]

    return {'train': train_samples,
            'val': val_samples,
            'test': test_samples}


def _convert_dataset(samples, filepath):
    num_paths = len(samples)
    print('Images to process: {}'.format(num_paths))

    origin_image_pl = tf.placeholder(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, None], dtype=tf.uint8)
    seg_image_pl = tf.placeholder(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, None], dtype=tf.uint8)

    origin_image_patches, seg_image_patches = preprocess_input.preprocess_input(origin_image_pl,
                                                                                seg_image_pl,
                                                                                origin_size=513,
                                                                                seg_size=129)

    num_examples = 0
    writer = tf.python_io.TFRecordWriter(filepath)
    with tf.Session() as sess:
        for i in range(num_paths):
            print('Process image: {}'.format(i + 1))

            origin_image = build_data.load_image(samples[i][0], type='JPG')

            seg_image = build_data.load_image(samples[i][1], type='PNG')
            seg_image = preprocess_input.map_to_classes(seg_image)

            origin_image_patches_res, seg_image_patches_res = sess.run([origin_image_patches, seg_image_patches],
                                                                       feed_dict={origin_image_pl: origin_image,
                                                                                  seg_image_pl: seg_image})

            for j in range(origin_image_patches_res.shape[0]):
                # visualization.visualize_segmentation(origin_image_patches_res[j], seg_image_patches_res[j])

                example = build_data.image_seg_to_tfexample(origin_image_patches_res[j], seg_image_patches_res[j])
                writer.write(example.SerializeToString())

                num_examples += 1
    writer.close()
    print('Number of train examples: {}'.format(num_examples))


def main(unused_argv):
    file_paths = _get_files(FLAGS.images_folder)
    splits = _split_dataset(file_paths, parts=FLAGS.splits)
    _convert_dataset(splits[FLAGS.split_part], FLAGS.output_file)


if __name__ == '__main__':
    flags.mark_flags_as_required(['images_folder', 'split_part', 'output_file'])
    tf.app.run()

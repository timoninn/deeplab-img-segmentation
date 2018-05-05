import tensorflow as tf

from core import model
from dataset import build_data


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


def main():
    iterator = _get_iterator(['../tmp/train.tfrecord'])
    _calculate_dec_outputs(iterator, '../tmp/train_prelogits.tfrecord')


if __name__ == '__main__':
    main()

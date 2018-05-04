import tensorflow as tf
import dataset.utils as dutils
import core.model as model
import core.preprocess_utils as cutils

worker_index = 0

deeplab = model.DeepLab()

dataset = tf.data.TFRecordDataset(['../tmp/val.tfrecord'])
dataset = dataset.shard(num_shards=10, index=worker_index)
dataset = dataset.map(dutils.parse_tfexample_to_image_seg)
dataset = dataset.repeat(1)
dataset = dataset.batch(1)
dataset = dataset.prefetch(2)

iterator = dataset.make_one_shot_iterator()
origin_image, seg_image = iterator.get_next()

# origin_image = cutils.resize_imgs(origin_image,
#                                   input_size=513)
#
# seg_image = cutils.resize_imgs(seg_image,
#                                input_size=129)

writer = tf.python_io.TFRecordWriter('../tmp/val_prelogits_10_{}.tfrecord'.format(worker_index))
with tf.Session() as sess:
    while True:
        try:
            origin_image_res, seg_image_res = sess.run([origin_image, seg_image])

            dec_output = deeplab.run_decoder(origin_image_res[0])
            print(dec_output.shape)
            print(seg_image_res.shape)

            example = dutils.decoder_seg_to_tfexample(dec_output[0], seg_image_res[0])
            writer.write(example.SerializeToString())

        except tf.errors.OutOfRangeError:
            print('Finish')
            break

writer.close()

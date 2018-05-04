import tensorflow as tf
import glob
import dataset.preprocess_dataset as preprocess
import dataset.utils as dutils
import core.preprocess_utils as cutils


def main(origin_paths, seg_paths, filepath):
    num_paths = len(origin_paths)
    print('Images to process: {}'.format(num_paths))

    origin_image_pl = tf.placeholder(shape=[None, 2710, 3384, None], dtype=tf.uint8)
    seg_image_pl = tf.placeholder(shape=[None, 2710, 3384, None], dtype=tf.uint8)

    origin_image_patches = preprocess.extract_patches(origin_image_pl)
    seg_image_patches = preprocess.extract_patches(seg_image_pl)

    origin_image_patches = cutils.resize_imgs(origin_image_patches,
                                              input_size=513)

    seg_image_patches = cutils.resize_imgs(seg_image_patches,
                                           input_size=129)

    writer = tf.python_io.TFRecordWriter(filepath)
    with tf.Session() as sess:
        for i in range(num_paths):
            print('Process image: {}'.format(i + 1))

            origin_image = dutils.load_image(origin_paths[i], type='JPG')

            print(origin_paths[i])

            seg_image = dutils.load_image(seg_paths[i], type='PNG')
            seg_image = preprocess.map_to_classes(seg_image)

            print(seg_paths[i])

            origin_image_patches_res, seg_image_patches_res = sess.run([origin_image_patches, seg_image_patches],
                                                                       feed_dict={origin_image_pl: origin_image,
                                                                                  seg_image_pl: seg_image})

            for j in range(9):
                # preprocess.visualize_segmentation(origin_image_patches_res[j], seg_image_patches_res[j])

                example = dutils.image_seg_to_tfexample(origin_image_patches_res[j], seg_image_patches_res[j])
                writer.write(example.SerializeToString())

    writer.close()


# seg_images_path = '../data/train_label_batch/*.png'
# origin_images_path = '../data/train_color_batch/*.jpg'

seg_images_path = '../data/main/seg/*.png'
origin_images_path = '../data/main/origin/*.jpg'

seg_images = glob.glob(seg_images_path)
origin_images = glob.glob(origin_images_path)

seg_images = sorted(seg_images)
origin_images = sorted(origin_images)

assert len(origin_images) == len(seg_images), 'Number of examples should match'

num_examples = len(origin_images)
print('Number of examples: {}'.format(num_examples))

train_origin = origin_images[0:int(0.6 * num_examples)]
train_seg = seg_images[0:int(0.6 * num_examples)]

val_origin = origin_images[int(0.6 * num_examples):int(0.8 * num_examples)]
val_seg = seg_images[int(0.6 * num_examples):int(0.8 * num_examples)]

test_origin = origin_images[int(0.8 * num_examples):]
test_seg = seg_images[int(0.8 * num_examples):]

# main(train_origin, train_seg, '../tmp/train.tfrecord')
main(val_origin, val_seg, '../tmp/val.tfrecord')

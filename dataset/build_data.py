import tensorflow as tf
import glob
import dataset.preprocess_dataset as preprocess
import dataset.utils as dutils


def main(origin_paths, segmentation_paths, filepath):
    writer = tf.python_io.TFRecordWriter(filepath)

    with tf.Session() as sess:
        for i in range(len(origin_paths)):
            segm_image = dutils.load_image(segmentation_paths[i], type='PNG')
            origin_image = dutils.load_image(origin_paths[i], type='JPG')

            segm_image = preprocess.map_to_classes(segm_image)

            origin_image_patches = preprocess.extract_patches(origin_image)
            segm_image_patches = preprocess.extract_patches(segm_image)

            origin_image_patches, segm_image_patches = sess.run([origin_image_patches, segm_image_patches])

            for j in range(9):
                # preprocess.visualize_segmentation(origin_image_patches[j], segm_image_patches[j])
                example = dutils.image_seg_to_tfexample(origin_image_patches[j], segm_image_patches[j])
                writer.write(example.SerializeToString())

    writer.close()


label_path = '../data/train_label_batch/*.png'
color_path = '../data/train_color_batch/*.jpg'

labels = glob.glob(label_path)
colors = glob.glob(color_path)

labels = sorted(labels)
colors = sorted(colors)

assert len(colors) == len(labels), 'Number of examples should match'

num_examples = len(colors)
print(num_examples)

train_labels = labels[0:int(0.6 * num_examples)]
train_colors = colors[0:int(0.6 * num_examples)]

val_labels = labels[int(0.6 * num_examples):int(0.8 * num_examples)]
val_colors = colors[int(0.6 * num_examples):int(0.8 * num_examples)]

test_labels = labels[int(0.8 * num_examples):]
test_colors = colors[int(0.8 * num_examples):]

# main(train_colors, train_labels, '../tmp/train.tfrecord')
main(val_colors, val_labels, '../tmp/val.tfrecord')

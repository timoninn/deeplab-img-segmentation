import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

LABEL_NAMES = np.array(['backgound', 'car', 'motorbicycle', 'bicycle', 'person', 'truck', 'bus', 'tricycle'])


def create_clolormap():
    """
    Create colormap used in PASCAL VOC segmentation benchmark.

    :return: np.array of shape [256, 3].
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def show_patches(patches):
    """
    Show 9 patches.

    :param patches: Patches with shape [9, height, width, num_channels].
    """
    plt.figure(figsize=(7, 7))
    for i in range(9):
        grid = int('33{}'.format(i + 1))
        plt.subplot(grid)
        plt.imshow(patches[i])
        plt.axis('off')

    plt.show()


def visualize_segmentation(image, seg_map):
    """
    Visualize segmentation result.

    :param image: Original image with shape [height, width, num_channels].
    :param seg_map: Segmentation map with shape [height, width, 1].
    """

    def _label2color(label):
        colormap = create_clolormap()
        return colormap[label]

    plt.figure(figsize=(20, 10))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[7, 7, 7, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')

    seg_map = np.squeeze(seg_map)
    color_seg_map = _label2color(seg_map)
    plt.subplot(grid_spec[1])
    plt.imshow(color_seg_map)
    plt.axis('off')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(color_seg_map, alpha=0.8)
    plt.axis('off')

    unique_labels = np.unique(seg_map)
    unique_colors = _label2color(unique_labels)
    unique_colors = np.expand_dims(unique_colors, 1)

    ax = plt.subplot(grid_spec[3])
    plt.imshow(unique_colors)
    ax.yaxis.tick_right()
    plt.yticks(range(unique_labels.shape[0]), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    plt.show()

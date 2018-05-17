Deeplab Image Segmentation

###Convert dataset to TFRecord

```bash
# From deeplab_img_segmentation/dataset
python3 build_cvpr_data.py \
    --images_folder ../data/demo/ \
    --splits 0.3 \
    --splits 0.2 \
    --splits 0.2 \
    --split_part train \
    --output_file ../tmp/train.tfrecord
```

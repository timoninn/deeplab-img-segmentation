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
    --output_file ../tmp/demo_train.tfrecord
```

###Precalculate decoder outputs
```bash
# From deeplab_img_segmentation/train
python3 preprocess_train.py \
    --data_path ../tmp/demo_train.tfrecord \
    --output_file ../tmp/demo_train_prelogits.tfrecord
```

### Train model
```
# From deeplab_img_segmentation/train
python3 train.py \
    --prelogits_path ../tmp/demo_train_prelogits.tfrecord \
    --train_logdir ../tmp/demo_train_logdir/ \
    --num_steps 100
```
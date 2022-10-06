## 1. REVEAL dataset
- REVEAL is a database for Retinal arteriole and venule anlysis which includes three sets of images of different image quality and diabetic retinopathy signs. We collated 20 images and labels from the REVEAL dataset (incorporating arteriovenous labels into one class only for distinguishing vessels and background) to train the vessel segmentation model for diabetic fundus images.
## 2. Dataset structure
```bash
.
├── train
│    ├── image_zoom_hd
│    │    └── *.jpg
│    └── label_zoom_hd
│         └── *.jpg
└── val
     ├── image_zoom_hd
     │    └── *.jpg
     └── label_zoom_hd
          └── *.jpg
```
## 3. Train the vessel segmentation model
- For training on REVEAL:
```bash
python terminal.py --vessel_seg=1 --vs_step=train --vs_batch_size=2 --vs_target_size=512 --vs_max_epoch=200
```
- The weight file of trained model will be save in [./results](results).

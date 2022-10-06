** Please decompress the datasets here!

** 1. Dataset description
- **IDRiD**

The [IDRiD dataset](https://doi.org/10.3390/data3030025) is provided by 2018 ISBI grand challenge on diabetic retinopathy segmentation and grading, which consists of 81 color fundus images and pixel-level annotations of four types of lesions. Of these 81 images, all images contain EX and MA, a set of 80 images contain HE, and 40 images contain SE. The images have a resolution of 4288×2848 pixels. The dataset is split to 54 training samples and 27 test samples by the organizer.
- **e-ophtha**

The [e-ophtha dataset](https://doi.org/10.1016/j.media.2014.05.004) consists of two subsets, including e-ophtha EX and e-ophtha MA. The e-ophtha EX subset consists of 82 images, of which 47 images contain EXs and 35 images are normal images. The e-ophtha MA subset consists of 381 images, of which 148 images contain MAs and the other 233 images are normal. The image resolutions range from 1504 × 1000 pixels to 2544 × 1696 pixels. We selected the 21 images containing both EXs and MAs from the e-ophtha dataset. We used 14 images for training and the rest 7 images for test.

** 2. Dataset structure
```bash
.
├── IDRiD
│    ├── image
│    │    ├── train
│    │    │    └── *.jpg
│    │    └── test
│    │         └── *.jpg
│    └── label
│         ├── train
│         │    ├── EX
│         │    │    └── *.tif
│         │    ├── HE
│         │    │    └── *.tif
│         │    ├── MA
│         │    │    └── *.tif
│         │    └── SE
│         │         └── *.tif
│         └── test
│              └── ...
│
└── e_ophtha
     ├── image
     │    ├── train
     │    │    └── *.png
     │    └── test
     │         └── *.png
     └── label
          ├── train
          │    ├── EX
          │    │    └── *.png
          │    └── MA
          │         └── *.png
          └── test
               └── ...
```

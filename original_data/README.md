- Please decompress the datasets here!
** 1. Dataset description
- IDRiD
- e_ophtha
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

# hpatches-features
This repository contains features (ORB, SIFT, LIFT, SuperPoint, D2-Net) extracted on HPatches Dataset (full images).

## How to extract keypoints and descriptors from these files?
`features` is a pickle file.
Let's say you want to extract ORB keypoints and descriptors.   
You can do it in python3 as follows:

```bash
$ python3
>>> import numpy as np
>>> import pickle
>>> with open('orb/features','rb') as f:
...     file = pickle.load(f)
>>> len(file) # there are 696 images in HPatches Dataset.
696
>>> img0_name = file[0][0]
>>> img0_name
'i_pinard'
>>> img0_file = file[0][1]
>>> img0_file 
'1.ppm'
>>> img0_shape = file[0][2]
>>> img0_shape
(600, 800)
>>> img0_keypoints = file[0][3]   # a np array of keypoints
>>> img0_descriptors = file[0][4] # a np array of descriptors
>>> # Play with ORB Features
```

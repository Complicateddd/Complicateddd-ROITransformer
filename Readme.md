# Huojianjun Competition

- ## Preliminary Ranking  A:42/296  B:31/163

## 1. Baseline: ROITransfomer:

https://github.com/dingjiansw101/AerialDetection



## 2. This Repo:

- [x] **Support Rotated Boxes detection.**

- [x] **Support draw origin ground truth rotated box.(draw_groundtruth.py)**

- [x]  **Support common image detection but not only supports large images of Dota dataset.  (demo_image.py)**

- [x]  **Support random crop/ random rotated  [Date augment)] (mmdet/datasets/rotate_aug.py extra_aug.py)**

- [x]  **Support Res2Net [Backbone] (Need the latest mmcv version and rename to mmcv1 ) (mmdet/models/backbones/res2net.py res2.py)**



### 3. Usage:

#### Environment:

1. Refer to ./**INSTALL.md** firstly
2. Install ./requirements.txt 
3. This repo need version of PyTorch (Nightly 1.3.0)
4. mmcv 0.2.16
5. To use res2net, you need to install the latest mmcv and rename the whole file name of mmcv to mmcv1
6. The other installation process you can refer to the baseline repo in  https://github.com/dingjiansw101/AerialDetection and check with my environment in ./package.txt

#### Dataet:

Competion dataset: 

1. Data form : 

   You need to set your own images and labels as follow.

   --DataSet

   ​	--images

   ​		1.jpg

   ​		2.jpg

   ​		...

   ​    --labelTxt

   ​	    1.txt

   ​	    2.txt

   ​	    ...

2. labels format:
	[class id  x1 y1 x2 y2 x3 y3 x4 y4]	

   4 586 459 577 441 646 410 654 428

   5 330 389 317 355 308 359 317 393

   ....

   

#### Train && Inference:

1. Modify and run the .DOTA_devkit/data2COCO.py to get .json file

2. Modify the category in ./mmdet/datasetsmydataset.py

3. Modify the config information in ./configs/Huojianjun/faster_rcnn_RoItrans_r101_fpn_anchors.py

4. Run the muti gpu train:

   ```python
   ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
   ```

4. Inference of image:

   ```python
   python demo_images.py
   ```

5. The more command you can refer to 

   https://github.com/dingjiansw101/AerialDetection

   and mmdetection usage:

   https://github.com/open-mmlab/mmdetection

   # Detection result:

   ![ship](https://raw.githubusercontent.com/Complicateddd/Complicateddd-ROITransformer/master/demo/demo.png)

##### To do:

- [ ] Cascade ROITransfomer
- [ ] Model Ensembl
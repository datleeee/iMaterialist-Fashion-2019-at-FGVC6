# Fine-grained segmentation task for fashion and apparel
**Problem:** Recognize apparel products and associated attributes from pictures.

![](https://s3.amazonaws.com/ifashionist/Kaggle/Kaggle3.jpg)

**Data**:
* 45.2k train - Images with or without fine-grained segmentations 
* 3193 test - Images with fine-grained segmentations
* We must segment and classify the images in test set

**Columns**:
* ImageId - unique Id of an image
* EncodedPixels - masks in run-length encoded format
* ClassId - class id for this mask

**Evaluation**: Mean average precision at different *Intersection over Union (IoU)* thresholds. 
IoU = (Area of overlap) / (Area of union)

## 1st place solution
Base on this paper: https://arxiv.org/abs/1901.07518

**Model:**
* [Mmdetection](https://github.com/open-mmlab/mmdetection): Hybrid Task Cascade with ResNeXt-101-64x4d-FPN backbone
* Has a metric Mask mAP = 43.9 on COCO dataset
* SOTA for instance segmentation

**Validation:**
* Split 450 samples from training set
* Scikit-learn cross validators for [iterative stratification](https://github.com/trent-b/iterative-stratification) of multilabel data

**Preprocessing:**
 * Light augmentations:  [albumentations](https://github.com/albumentations-team/albumentations) library
 * Multi-scale training: in each iteration
 * Short edge: randomly sampled from [600, 1200]
 * Long edge: fixed at 1900
![](https://raw.githubusercontent.com/amirassov/kaggle-imaterialist/master/figures/preproc.png)

**Training:**
* Pre-trained from COCO
* Optimizer: SGD(lr=0.03, momentum=0.9, weight_decay=0.0001)
* Batch_size: 16 = 2 images per gpu x 8 gpus tesla V100
* Learning rate:
 (if iterations < 500: lr = warmup(warmup_ratio=1 / 3) if epochs == 10: lr = lr ∗ 0.1 if epochs == 18: lr = lr ∗ 0.1 if epochs > 20: stop)
* Training time: 3 days

**Parameter tuning:**
After 12th epoch with default parameters: metric = **0.21913**
Using validation data for tuning:
* score_thr = 0.5
* nms = {type: 'nms' , iou_thr: 0.3}
* max_per_img = 100
* mask_thr_binary = 0.45
Improved metric: **0.21913** --> **0.30011**

**Test time augmentation:**
* 3 scales, horizontal flip:
    * (1000,1600)
    * (1200, 1900)
    * (1400, 2200)
* TTA scheme for Mask R-CNN, implemented in mmdetection library
* Improved metric **0.30011** --> **0.31074**
![](https://raw.githubusercontent.com/amirassov/kaggle-imaterialist/master/figures/tta.png)

**Ensemble:**
* Ensemble 3 best checkpoints of model
* Ensemble scheme similar to TTA
* Improved metric **0.31074** --> **0.31626**
![](https://raw.githubusercontent.com/amirassov/kaggle-imaterialist/master/figures/ensemble.png)

**Attributes:**
* Didn't use attributes: difficult to predict
* Deleted classes with attributes: {0,1,2,3,4,5,6,7,8,9,10,11,12} 
* Improved metric **0.31626** --> **0.33511**

## 3rd place solution

**Preprocessing:** 
* Minsize: (800,...960)
* Maxsize <= 1600

**Model:**
* Facebook repo - Mask-RCNN x-101
* Mmdetection repo - Hybrid Task Cascade with X-101-64x4d-FPN backbone and c3-c5 DCN

**TTA-4:**
* hflip
* Different sizes (minsize=800 and minsize=960)

**Attributes:**
* Dropped all predictions for categores 0--12

## 18th place solution

**Model**: r101 Mask-RCNN 1024x1024

**Ansebmle prediction** from multiple snapshots of r101 Mask-RCNN

[Classfication Pipeline](https://github.com/musket-ml/classification_training_pipeline): Train several **multiclass** cassfiers

[Segmentation Pipeline](https://github.com/musket-ml/segmentation_training_pipeline): Train lots of **segmentation** networks at least one per class --> refine masks from mask-rcnn (**main source for improvement**)

## Approaches work best:
* [mmdetection](https://github.com/open-mmlab/mmdetection) for Mask-RCNN: Hybrid Task Cascade with X-101-64x4d-FPN backbone 
* [iterative stratification](https://github.com/trent-b/iterative-stratification): cross validation for multilabel data
* [albumentations](https://github.com/albumentations-team/albumentations): light data augmentation

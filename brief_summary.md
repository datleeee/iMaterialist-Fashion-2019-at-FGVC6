# Fine-grained segmentation task for fashion and apparel
**Problem:** Recognize apparel products and associated attributes from pictures.
**Data**:
* train - Images with or without fine-grained segmentations 
* test - Images with fine-grained segmentations
* We must segment and classify the images in test set

**Columns**:
* ImageId - unique Id of an image
* EncodedPixels - masks in run-length encoded format
* ClassId - class id for this mask

**Evaluation**: Mean average precision at different *Intersection over Union (IoU)* thresholds
![math](/downloads/math.png)

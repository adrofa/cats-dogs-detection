# Code Description
`modules` directory:
* `data.py` - contains `Dataset` class for PyTorch and some functions for basic data operations;
* `custom_models` - contains architectures created by myself from scratch;
    * `models.py` - architectures itself;
    * `layers.py` - layers;
* `run` - contains scripts for training and inference:
    * `train_clasifier.py` - for training classifier only architectures;
    * `train_detector.py` - for training classifier + detector architectures;
    * `versions` - contains versions of models, augmentation, loss functions, optimizers and
      lr-schedulers, which will be called in `train_*.py` scripts. 
* `scripts` - contains scripts to be rarely called:
    * `cross_validation_split.py` - splits `train` dataset into folds;
    * `dataset_df.py` - collects dataset information (annotation, image sizes etc.)
    * `image_normalization.py` - estimates images mean and std for normalization.
* `notebooks` - collects created Jupyter Notebooks:
    * `Initial_EDA.ipynb` - checks how train dataset matches valid dataset;
    * `colab` - contains notebooks run in Google Colab with `run/train_*.py` scripts.
      <u>Notebooks naming is inline with Tested Hypothesis versions described below</u>. 

# Tested Hypothesis
**Loss function** for training classifier+detector model I used combined loss function:
* **binary cross-entropy** - for classification;
* **smooth-L1 x weight** - for bounding boxes (bounding boxes were normalized to range \[0, 1]).

Before Version 1 (described below) I run model for several times to find optimal **weight**
for smooth-L1 value and started from **75**

Also I made couple experiments with the pretrained **Xception**,
but **EfficientNet-B0** showed better results, hence I continue with this backbone.
*Both pretrained models are from [`timm`](https://pypi.org/project/timm/) package.*

*For more parameters versions details consider `run/versions/<parameter>_version`.* 

## Transfer Learning
Experiments performed with pretrained models are described below. 

### Version 1 | IoU: 0.427
1. Take EfficientNet-B0 model pretrained on imagenet;
2. Replace final FC layer with a custom one;
3. Train until early stopping.
* `model_version: v1`
* `augmentation_version: v1`
* `criterion_version: v1`
* `optimizer_version: adam_v1`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v1/config.json)

Results:
![v1](/output/models/detector/v1/progress.png)

### Version 2 | IoU: 0.425
1. Take model (with the best valid loss) from Version 1;
2. Unfreeze last 2 CONV layers;
3. Increase bbox weights in Loss function 75 -> 100 (model does good classification,
   but bad bboxes preds);
4. Decrease LR in optimizer
5. Train until early stopping.
* `model_version: v2`
* `model_weiights: version_v1`
* `augmentation_version: v1`
* `criterion_version: v2`
* `optimizer_version: adam_v2`
* `scheduler_version: rop_v2`
* [full config](/output/models/detector/v2/config.json)

Results:
* unfreezing additional layers returned no positive results
* I suppose, that classifier-head was to heavy and found its local minimum
![v2](/output/models/detector/v2/progress.png)
  
### Version 3 | IoU: 0.395
1. Take EfficientNet-B0 model pretrained on imagenet;
2. Replace final FC layer with a custom one (lighter than in version 1);
3. Train until early stopping.
* `model_version: v3`
* `augmentation_version: v1`
* `criterion_version: v1`
* `optimizer_version: adam_v1`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v3/config.json)

Results:
![v3](/output/models/detector/v3/progress.png)


### Version 4 | IoU: 0.401
1. Take model (with the best valid loss) from Version 3;
2. Unfreeze last 2 CONV layers;
3. Increase bbox weights in Loss function 75 -> 100 (model does good classification,
   but bad bboxes preds);
4. Decrease LR in optimizer
5. Train until early stopping.
* `model_version: v4`
* `model_weiights: version_v3`
* `augmentation_version: v1`
* `criterion_version: v2`
* `optimizer_version: adam_v2`
* `scheduler_version: rop_v2`
* [full config](/output/models/detector/v4/config.json)

Results:
![v4](/output/models/detector/v4/progress.png)

### Version 5 | IoU: 0.714
I performed several local tests on Version 4:
* with different LRs;
* w/o separate FC layer pretraining;
* higher bounding boxes loss weight.

Tests showed, that model converges much faster with higher LR and
a separate FC layer pretraining is not needed.

Also, I looked on predicted bounding boxes and images and noticed that
there are 2 groups of images:
* "horizontal" (with > height)
* "vertical" (height > width).

Previously I was resizing images to fixed square (e.g. 256x256). Hence a net need to learn features
from 2 different. To fix this issues I replaced basic resizing to resizing by the longest side +
padding.

1. Take EfficientNet-B0 model pretrained on imagenet;
2. Unfreeze last 2 CONV layers;
3. Increase bbox weights in Loss function to 1_000;
4. Increase LR to 0.01
5. Train until early stopping.
* `model_version: v4`
* `model_weiights: None`
* `augmentation_version: v2`
* `criterion_version: v3`
* `optimizer_version: adam_v3`
* `scheduler_version: rop_v2`
* [full config](/output/models/detector/v5/config.json)

Results:
![v5](/output/models/detector/v5/progress.png)

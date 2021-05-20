# Code Description
`modules` directory:
* `data.py` - contains `Dataset` class for PyTorch and some functions for basic data operations;
* `custom_models` - contains architectures created by myself from scratch;
    * `models.py` - architectures itself;
    * `layers.py` - layers;
* `run` - contains scripts for training and inference:
    * `train_clasifier.py` - for training classifier only architectures;
    * `train_detector.py` - for training classifier + detector architectures;
    * `inference.py` - for generating class and bounding boxes predictions
      (also, TOP-3 best and worst bounding boxes predictions examples);
    * `versions` - contains versions of models, augmentation, loss functions, optimizers and
      lr-schedulers, which will be called in `train_*.py` scripts. 
* `scripts` - contains scripts to be rarely called:
    * `cross_validation_split.py` - splits `train` dataset into folds;
    * `dataset_df.py` - collects dataset information (annotation, image sizes etc.)
    * `image_normalization.py` - estimates images mean and std for normalization.
* `notebooks` - contains created Jupyter Notebooks:
    * `Initial_EDA.ipynb` - checks how train dataset matches valid dataset;
    * `colab` - contains notebooks run in Google Colab with `run/train_*.py` scripts.
      <u>Notebooks naming is inline with Tested Hypothesis versions described below</u>.

## Transfer Learning
Experiments performed with pretrained models are described below.

### Results
Final solution [`/output/inference/v1`](/output/inference/v1):
<br/>`mIoU 85%, classification accuracy 99%, 8.99ms, 2686 train, 400 valid`
* mean-IoU: 85%
* accuracy: 99%
* inference time pepr image: 8.17ms
* train size: 2686 *(cross_validation_split-v0 fold-0)*
* valid size: 400

#### Bounding Box Predictions Examples
##### TOP3 Best Predictions
![top3_best](/output/inference/v1/top3_best.png)
##### TOP3 Worst Predictions
![top3_worst](/output/inference/v1/top3_worst.png)

### Tested Hypothesis
While training the model I wasn't "looking" at valid dataset and split the train dataset
in 2 parts: 90% - train and 10% - valid. 

**Loss function** for training classifier+detector model I used combined loss function:
* **binary cross-entropy** - for classification;
* **smooth-L1 x weight** - for bounding boxes (bounding boxes were normalized to range \[0, 1]).

Before Version 1 (described below) I run model for several times to find optimal **weight**
for smooth-L1 value and started from **75**

Also I made couple experiments with the pretrained **Xception**,
but **EfficientNet-B0** showed better results, hence I continue with this backbone.
*Both pretrained models are from [`timm`](https://pypi.org/project/timm/) package.*

*For more details regarding run parameters: model, optimizer augmentation, criterion, scheduler
consider `run/versions/<parameter>`.* 

#### Version 1 | IoU<i>@best_loss</i>: 0.427
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

#### Version 2 | IoU<i>@best_loss</i>: 0.425
1. Take model (with the best valid loss) from Version 1;
2. Unfreeze last 2 CONV layers;
3. Increase bbox weights in Loss function 75 -> 100 (model does good classification,
   but bad bboxes preds);
4. Decrease LR in optimizer
5. Train until early stopping.
* `model_version: v2`
* `model_weights: version_v1`
* `augmentation_version: v1`
* `criterion_version: v2`
* `optimizer_version: adam_v2`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v2/config.json)

Results:
* unfreezing additional layers returned no positive results
* I suppose, that classifier-head was to heavy and found its local minimum
![v2](/output/models/detector/v2/progress.png)
  
#### Version 3 | IoU<i>@best_loss</i>: 0.395
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


#### Version 4 | IoU<i>@best_loss</i>: 0.401
1. Take model (with the best valid loss) from Version 3;
2. Unfreeze last 2 CONV layers;
3. Increase bbox weights in Loss function 75 -> 100 (model does good classification,
   but bad bboxes preds);
4. Decrease LR in optimizer
5. Train until early stopping.
* `model_version: v4`
* `model_weights: version_v3`
* `augmentation_version: v1`
* `criterion_version: v2`
* `optimizer_version: adam_v2`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v4/config.json)

Results:
![v4](/output/models/detector/v4/progress.png)

#### Version 5 | IoU<i>@best_loss</i>: 0.714
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
* `model_weights: None`
* `augmentation_version: v2`
* `criterion_version: v3`
* `optimizer_version: adam_v3`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v5/config.json)

Results:
![v5](/output/models/detector/v5/progress.png)


#### Version 6 | IoU<i>@best_loss</i>: 0.811
Here I wil:
1. take Version 5 model;
2. unfreeze 1 more layer (total: last 3 Conv layers).

* `model_version: v5`
* `model_weights: version_v5`
* `augmentation_version: v2`
* `criterion_version: v3`
* `optimizer_version: adam_v3`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v6/config.json)

Results:
![v6](/output/models/detector/v6/progress.png)

### Version 7 | IoU: 0.796
Here I wil:
1. take Version 6 model;
2. unfreeze 1 more layer (total: last 4 Conv layers).

* `model_version: v6`
* `model_weiights: version_v6`
* `augmentation_version: v2`
* `criterion_version: v3`
* `optimizer_version: adam_v3`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v7/config.json)

Results:
![v7](/output/models/detector/v7/progress.png)

### Version 8 | IoU: 0.838
In Version 7 IoU dropped to ~0.6 (from ~0.8) after 1st epoch.
In this version I will try to fix it with LR reduction.

(This run was manually stopped after 45th epoch)

* `model_version: v6`
* `model_weiights: version_v6`
* `augmentation_version: v2`
* `criterion_version: v3`
* `optimizer_version: adam_v4`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v7/config.json)

Results:
![v8](/output/models/detector/v8/progress.png)

#### Version 9 | IoU<i>@best_loss</i>: 0.847
Version 8 was converging well, but slowly. I believe that Version 7 (with too high LR) failed
because ADAM-optimizer was started from scratch.

In Version 9 I will use "pretrained" optimizer from Version 8 and increase LR.

* `model_version: v6`
* `model_weights: version_v8`
* `augmentation_version: v2`
* `criterion_version: v3`
* `optimizer_version: adam_v5`
* `optimizer_weights: version_v8`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v7/config.json)

Results:
![v9](/output/models/detector/v9/progress.png)

#### Version 10 | IoU<i>@best_loss</i>: 0.85
Here I wil:
1. take Version 9 model;
2. unfreeze next Conv layer (total: last 5 Conv layers);
3. LR: 0.0001

* `model_version: v6`
* `model_weights: version_v9`
* `augmentation_version: v2`
* `criterion_version: v3`
* `optimizer_version: adam_v4`
* `optimizer_weights: None`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v10/config.json)

Results:
![v10](/output/models/detector/v10/progress.png)

#### Version 11 | IoU<i>@best_loss</i>: 0.847
Here I wil:
1. try to train model from Version 10 with a higher LR;
2. return old augmentation (w/o padding, just resizing),
   otherwise I need to produce additional code for bounding boxes inference. 

* `model_version: v6`
* `model_weights: version_v10`
* `augmentation_version: v1`
* `criterion_version: v3`
* `optimizer_version: adam_v3`
* `optimizer_weights: version_v10`
* `scheduler_version: rop_v1`
* [full config](/output/models/detector/v11/config.json)

Results:
![v11](/output/models/detector/v11/progress.png)
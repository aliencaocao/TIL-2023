# DSTA Brainhack Today-I-Learnt AI Hackathon 2023

#### Team: 10000SGDMRT
#### School: [Hwa Chong Institution Infocomm & Robotics Society (HCIRS)](https://github.com/hcirs) Machine Learning section
#### Achievement: Champion in Advanced category, with a perfect score in the final match

### Team members
* [Billy Cao](https://github.com/aliencaocao) (L): ASR/SpeakerID/OD/ReID/Robot
* [Marcus Wee](https://github.com/Marcushadow): ASR
* [Ho Wing Yip](https://github.com/HoWingYip): ASR/SpeakerID/OD/ReID/Robot
* [Huang Qirui](https://github.com/hqrui): Robot
* [Alistair Cheong](https://github.com/cheongalc): ASR/SpeakerID/OD/ReID/Robot

## Contents
<!-- TOC -->
  * [Introduction](#introduction)
    * [Qualifiers](#qualifiers)
    * [Qualifiers private leaderboard](#qualifiers-private-leaderboard)
    * [Finals](#finals)
    * [Finals leaderboard](#finals-leaderboard)
  * [ASR](#asr)
    * [Data Augmentation](#data-augmentation)
    * [Model](#model)
    * [Training](#training)
    * [Inference](#inference)
    * [Finals-specific tuning](#finals-specific-tuning)
  * [Object Detection](#object-detection)
    * [Data Augmentation](#data-augmentation-1)
    * [Model](#model-1)
    * [Training](#training-1)
    * [Inference](#inference-1)
    * [Finals-specific tuning](#finals-specific-tuning-1)
  * [Object Re-Identification (REID)](#object-re-identification-reid)
    * [Data Augmentation](#data-augmentation-2)
    * [Model](#model-2)
    * [Training](#training-2)
    * [Inference](#inference-2)
    * [Finals-specific tuning](#finals-specific-tuning-2)
  * [Speaker Identification](#speaker-identification)
    * [Data Preprocessing - denoising](#data-preprocessing---denoising)
    * [Data Augmentation](#data-augmentation-3)
    * [Model](#model-3)
    * [Training](#training-3)
    * [Inference](#inference-3)
    * [Finals-specific tuning](#finals-specific-tuning-3)
  * [Robot](#robot)
    * [Localization](#localization)
    * [Path planning](#path-planning)
    * [Movement](#movement)
<!-- TOC -->

## Introduction
This repository contains all the code used for our team at TIL 2023 except for model weights. If you would like the weights, you may email aliencaocao@gmail.com. 

TIL 2023 has 2 categories: Novice and Advanced. The advanced category are for teams with prior experience in AI/Robotics and/or have taken relevant courses in university. Our team was placed into this category as we won [TIL 2022](https://github.com/aliencaocao/TIL-2022).

The competition has 2 stages: qualifiers and finals.

### Qualifiers
In qualifiers, participants are given 2 tasks:
1. ASR: Speech to text on Singaporean accent dataset. For advanced category, the dataset has been mixed with noise and time masking.
2. CV: Single-class object detection of soft toys (plushie), chained with Object Re-identification on unique plushies (identifying same plushie from different images take from different angles). For advanced category, the dataset has been mixed with camera noise and white balance perturbation.

The combined ranking is determined by the average ranking of a team in each task, tie breaking using average absolute score.

### Qualifiers private leaderboard
* [Novice ASR](leaderboards/nov_asr.json)
* [Novice CV](leaderboards/nov_cv.json)
* [Novice Combined](leaderboards/nov_combined.json)
* [Advanced ASR](leaderboards/adv_asr.json)
* [Advanced CV](leaderboards/adv_cv.json)
* [Advanced Combined](leaderboards/adv_combined.json)

Team 10000SGDMRT obtained 3rd on ASR with Word Error Rate (WER) of 1.4049318056670998%; 1st on CV with mean average precision @ IoU=0.5 (mAP=0.5) of 0.9301437182295662. This gave us a combined score of 0.9580472000864476 and placing us top in the Qualifiers in Advanced category.

### Finals
In finals, participants are given additional 2 tasks:
1. Speaker Identification: Identify the speaker of a given audio clip. These audio clips are recorded by finalists and teams need to identify the team and member ID of the speaker. For advanced category, the dataset has been mixed heavily with noise. Training data is extremely limited at 1 sample per class, and validation set is also limited at 5 samples (only 5/32 classes are covered).
2. Integration of ASR, Object Detection, Object Re-identification and Speaker Identification to drive a DJI Robomaster robot around a maze. The robot has to automatically plan paths and navigate to task checkpoints, then perform these tasks correctly there.

The finals is split into 2 groups. First, round-robin is carried out within each group, then, top place of each group compete for 1st and 2nd place, while 2nd place from each group compete for 3rd and 4th place. Ranking within each group is determined by maze score, which is calculated by [10 * (correct OD+ReID tasks) + 5 * correct SpeakerID tasks]. The ASR task was used to determine whether the robot will receive the next task checkpoint or a detour checkpoint. Each run is capped to 10 minutes and teams in Advanced category has 5 checkpoints to clear. Time to finish all 5 checkpoints will be used as a tiebreaker.

Our team's final run: https://www.youtube.com/watch?v=zCUGX4jAcEk

### Finals leaderboard
Novice:

![Novice](leaderboards/Novice.png)

Advanced:

![Advanced](leaderboards/Advanced.png)

## Our solution
We will introduce our approaches below. Each part contains a finals-specific tuning section where we document things we have done to improve score during finals according to the rules.
## ASR
All code can be found in [ASR](ASR).
### Data Augmentation

* SpecAugment with feature masking probability of 0.3 and size of 10 samples, no time masking as it may cause spoken words to be masked. This is built in by Hugging Face Transformers
* [Audiomentations](https://github.com/iver56/audiomentations):
    1. HighShelfFilter(max_gain_db=6.0, p=0.3)
    2. LowShelfFilter(max_gain_db=6.0, p=0.3)
    3. TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2)
    4. BandPassFilter(p=0.3)

Audiomentation augmentations are determined by listening to and reverse-engineering test samples using frequency and spectrum analysis.

### Model
We used Meta's [wav2vec2-conformer-rel-pos-large-960h-ft](https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large-960h-ft), which is a Wav2Vec2-conformer model with relative position embedding finetuned on 960 hours of LibreSpeech dataset. This model was proposed in [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) table 3 and 4. It is best open-sourced model on LibriSpeech test-other dataset according to [Paperswithcode](https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-other). We chose test-other dataset as it contains more noise than test-clean which suits our test data more.

### Training
We unfroze the entire network (instead of the classification layers only) and finetuned on DSTA-provided 3750 samples of audio data. We trained for 30 epochs but took the best checkpoint (lowest val WER) at step 2337 (around 25 epoch), then resumed training from there for another 20 epochs at a lower learning rate.
* Optimizer: AdamW with lr=1e-4 for first 30 epoch run then continue with 1e-6 for another 20 epoch, beta1=0.9, beta2=0.98, weight decay=1e-4. Within each run, we linearly warmup lr from 0 for first 10% of steps, then do cosine decay for the rest of the steps until 0.
* Batch size: with usage of gradient accumulation, we had effective batch size of 32.
* Trained in FP16 with mixed precision.

Training log for the 2nd part (20 epoch @ lr=1e-6) can be found [here](ASR/wav2vec2-conformer/trainer_state.json).

Although we managed to obtain and preprocess the [IMDA National Speech Corpus](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus), our model failed to converage on these data, possibly due too low label quality, so we did not use them.

### Inference
In addition to normal inference, we use [language-tool-python](https://pypi.org/project/language-tool-python/) to correct spelling and grammar errors automatically, although it's not perfect, it improved WER slightly. This is needed as our model uses character-based tokenization (E.g. letter based) instead of word-based like Whisper.

Model ensemble via logit averaging was found to be bad for character-based tokenization, so we did not use it.

### Finals-specific tuning
The finals task is to recognize a digit from zero to nine in a given sentence. In other word, our model only has to get the digit part of the audio correct. Thus, we developed a fuzzy-retrival algorithm that uses string similarity to match and look for possible misspelled digits in the raw output. Again, this is only an issue as it is character-based tokenization.

We used [Levenshtein distance](https://pypi.org/project/python-Levenshtein/) as a metric of string similarity, and computed it across every word in the raw output against every digit, forming a 2D matrix. Then we take the globally most similar word and assume that it is the misspelled version of that closest digit. In the event where multiple words has the same similarity, it ignores some common words that can happen to have a high similarity with certain digits, like 'To' and 'Two', only taking them if no other word has the same level of similarity. The common words list used was ['TO', 'THE', 'THEY', 'HE', 'SHE', 'A', 'WE'].

Eventually, this algorithm was never activated once during finals as our model was good enough that it predicted everything perfectly. We had a 100% accuracy for ASR during finals.

## Object Detection
All code can be found in [CV/InternImage](CV/InternImage). The `CV` folder also contains some of our tried and ditched models:
* GroundingDINO: zero shot object detection guided by text prompt. We prompted 'plushie'. It has 0.345255479 mAP on bbox only (no REID).
* RT-DETR-X: novel fast and accurate YOLO-series replacement. We had highest mAP of 0.915573138 by this model combined with our best REID model, just 0.015 shy of our best using InternImage (and same REID model). We used implementation from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr). We tried various augmentations and none of them seemed to have helped.
* SAM: failed attempt to use zero-shot masks as bbox. We found that mask overlapping is a major issue and could not find a reliable NMS algorithm to handle them.
* yolov7 and yolov8: both underperformed RT-DETR, even with gaussian noise augmentations.

The other folders in `CV` are for ReID task. We will only explain our best performing model, [InternImage](https://github.com/OpenGVLab/InternImage).
### Data Augmentation
Default used in InternImage, including RandomFlip and AutoAug. See [our config](CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/cascade_internimage_l_fpn_3x_coco_custom.py).

### Model
We used InternImage-L with Cascade Mask R-CNN with 3x schd. It has 56.1mAP on COCO, 277M parameters and 1399GFLOPS. The backbone is pretrained on ImageNet-22K on a variable input size between 192 to 384px.

### Training
We used default hyperparameters except for batch size where we had to set to 1 for it to not OOM. We trained for 30 epochs but took the 12th epoch as it is the best checkpoint (val mAP@0.5-0.95 of 0.8652, mAP@0.5 of 1.0).

Since the model n uses Mask R-CNN, we had to process the bbox label given into mask labels too, by simply taking the bbox as a rectangle mask. The model does output both bbox and mask, so during inference we took the bbox output only.

Our training log is [here](CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/20230601_224318.log).

### Inference
By visualizing and manually checking through every single bbox on test set, we found that this model produces exceptionally high confidence score, often above 0.999. We found some false positives with relatively high confidence of 0.8+. Therefore, for qualifiers, we inferred with confidence threshold of 0.99, and take the top 4 confidence one if there are more than 4 (competition rules specified this).

### Finals-specific tuning
Although the model is very good at the task already, during our first day in finals, we found that it struggles to detect very small targets, especially a bee plushie, since the robot's camera is wide angle and the plushie itself is much smaller than average. It also has some false positives caused by the aruco codes on the floor. To counter this issue, we applied cropping to the robot camera view, by cropping 100px from top, left and right, and 280px from bottom, making the effective image size to be 900x520. This effectively zooms in and enlarge all the targets, while cropping away the floor most of the time. We also reduced confidence threshold to 0.9 as the it still tend to give lower confidence on smaller objects.

## Object Re-Identification (REID)
All code can be found in [CV/SOLIDER-REID](CV/SOLIDER-REID). This is a heavily modified fork from the [original repo](https://github.com/tinyvision/SOLIDER-REID). Specifically, we
* Added batched inference support
* Customized augmentations
* Added TIL Custom dataset
* Added plotting code for distance distribution
* Various bug fixes for loading and saving models

Nonetheless, we would like to thank the authors of the original repo for their work and their assistance in helping us debug our code.

SOLIDER-REID is a downstream task of a greater pretraining framework called SOLIDER, developed by Alibaba DaMo Academy, published in their paper [Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks](https://arxiv.org/abs/2303.17602).

This approach was the state-of-the-art (as of May 2023) on the largest person-reid dataset MSMT17. Although this is primarily for human-reid, we found no other 'generalized' object-reid benchmarks so we had to trust it and the scale of the dataset to be large enough to show the generalization ability of the model on cross-domain tasks. Turns out we were not wrong on this, and the model performed well on our task.

### Data Augmentation
We changed the default input size of the model from 384x128 to 224x224, reason being the plushie may not be a rectangle all the time like human, and the backbone (Swin-base) was pretrained on ImageNet-22k with input size 224x224.

We used the default augmentations from the original repo, which includes RandomHorizontalFlip(p=0.5), RandomCrop(size=224), RandomErasing(p=0.5, mode=pixel, max_count=1).

We added the following:
* GaussianNoise(probability=0.5, noise_strength=25). Our implementation can be found [here](CV/SOLIDER-REID/datasets/transforms.py). The aim is to emulate the camera ISO noise in our test set.
* Rotation Expansion. We hypothesize that unlike human who can only appear in an image standing up right (assuming no one is lying down or sitting), plushies can (and will) be placed in all sorts of directions, like up-side-down, lying horizontally etc. For the REID model to learn to be invariant to orientation of a plushie, we need to explicitly tell it to learn from different orientations. Thus, we rotated all test images by 90 degrees 3 times to cover the entire 360 degrees cycle, effectively 4x the dataset size. This has improved our leaderboard score slightly (as it was already very high) but we found it to became much more robust.

We tried a bunch of other augmentations and all of them decreased performance so I will not list them here, however they are still in the codebase.

We also tried to correct the white balance of images during train and test as they affect coloring of the plushies and could negatively affect the model. We used [Deep-White-Balance](https://github.com/mahmoudnafifi/Deep_White_Balance) and the results were visually satisfactory. See [here](CV/utils/awb exp) for sample images. Yet, this somehow significantly reduced our model performance, so we did not use it in the end. Our hypothesis is that the white balancing model actually causes different plushies to look a lot more similar (much more to the model's eye than to ours), and thus the model had a hard time differentiating them. We semi-verified this by plotting the distance distribution and seeing all of them are very close e.g. no clear separation. We also tried to use non-deep learning based apporaches like gray world or other algorithms in OpenCV and online, and none of them worked better than the DL approach visually.

### Model
We fine-tuned from the model pretrained on MSMT17 dataset as it is the largest. The model uses Swin-base transformer as backbone. We tried Swin-Large and found it underperformed slightly possibly due to overfitting.

### Training
Training this model proved to be especially tricky, as it tends to overfit quickly and this is hard to detect given the small validation set size (only 10 unique plushies). We had to run many times and made many submissions to find the way to pick the less-overfitting one. Specifically, we had to strike a balance between validation metrics like val mAP and training epochs. It has been shown repeatedly that anything above 92 for val mAP will be non-representative, meaning performance of a model getting mAP 92.0 vs 95.0 on val set may not follow the same trend. In fact, more than often they are opposite. We went for high epoch number (100+) combined with a relatively high mAP (93-94).

We modified the default hyperparameters through countless trial-and-error. Our final hyperparameters can be found in [config file](CV/SOLIDER-REID/TIL.yml). Similar to ASR, we took the 2-stage approach, where we first train the model with a higher initial LR of 5e-3 for 500 epoch, then we took the best performing one (based on heuristics mentioned above), which ended up being epoch 189. We then continued from epoch 189 for another 400+ epoch, with initial LR of 1e-4. We chose epoch 21 from this continued run.

* Optimizer: SGD with weight decay=1e-4, 3 warm up epoch and cosine decay LR.
* Batch size: 128
* Loss: average of Cross Entropy Loss and Triplet Loss. 
  * The Cross Entropy Loss is calculated only on training data where number of class is known (200). The model has 2 output heads, 1 is features (1024-dim), another is softmax (200 classes). During training, sum of both loss are used for back-propagation. During inference, only the features head is used to get raw features. This guides the model to converge and generate more useful feature.
  * The Triplet Loss is a customized one that uses harder example mining to maximize the loss's effectiveness. See [here](CV/SOLIDER-REID/loss/triplet_loss.py) for implementation.

Initial run of LR=5e-3 training log can be found [here](CV/SOLIDER-REID/archive models/log_SGD_500epoch_5e-3LR_expanded/initial_train_log.txt).

Continued run of LR=1e-4 training log can be found [here](CV/SOLIDER-REID/log_SGD_500epoch_continue_1e-4LR_expanded/continue_train_log.txt).

### Inference
We uses the [K-reciprocal Encoding for Re-ranking](https://arxiv.org/abs/1701.08398) algorithm and it improved mAP slightly for us. In order to find the right threshold, we plotted the distance distribution like this:

On validation set:

![distance distribution](CV/SOLIDER-REID/log_SGD_500epoch_continue_1e-4LR_expanded/21_val/reranking_separation_chart.png)

On test set:

![distance distribution](CV/SOLIDER-REID/log_SGD_500epoch_continue_1e-4LR_expanded/21_test/reranking_test_set_separation_chart.png)

Initially using the thresholds directly found at the minium point did not gave us a very high score. We suspect it is due to too many false positives, thus we assumed and limited the number of suspects per image to 1 (even though the competition rule did not specify) and used a more loose threshold to reduce false negatives. Doing so greatly improved our score by about 10mAP.

We also tried inferring on rotated image (4x90 degrees) and average their feature before calculating distance. This did not bring any improvement.

### Finals-specific tuning
For finals, we cannot use Reranking as the distance produced by RR changes with gallery size. In finals, we only have a gallery size of about 1-3 images, while in qualifiers we had 3417. Thus, we went to find the best threshold using euclidean distance based on leaderboard score, which turned out to be 1.05. However, this proved to be slightly off for finals, as we gathered data and analyzed the distance matrix produced during day 1's runs. Eventually, we reduced it slightly to be 1.0.

We initially had issues of false positives in our Object Detection and it kept detecting the floor and black tapes as a plushie. Interestingly those floor images ended up having very high similarity as the hostage in our REID model, causing us to predict many false positive for `hostage`. Applying cropping solved this issue.

We also realise that the query image given for finals is way higher resolution than the ones given during qualifiers, while the gallery image (crops of bbox from robot camera) is of much smaller resolution than the qualifier, partially due to robot's wide angle camera and far distance from the plushies.

For example, a query image from qualifiers are around the size 170x160, and gallery are around 120x100. While in finals, the hostage Mr Dastan, is 1275x1280, and most of the bbox we get is around 70x70.

Mr Dastan (1275x1280)

![Mr Dastan](Robot/data/imgs/HOSTAGE.jpg)

A bbox of a bee plushie (61x65)

![Bee](Robot/data/imgs/bee_box.png)

We hypothesis that this causes significant loss or instability in up/down scaling during resize operations, where images has to be resized to 224x224 for the REID model. We tried different resizing algorithms available in `OpenCV`, and they had shockingly large impact on our REID distance matrix:
 
Below are the comparison of distance matrix produced by our REID model using different resizing algorithms. This is calculated on bbox crops of various plushies captured by the robot against a [green dinosaur suspect](Robot/data/imgs/SUSPECT_2.jpg) and the hostage [Mr Dastan](Robot/data/imgs/HOSTAGE.jpg). The threshold is 1.0 so anything below 1.0 will be a positive. The optimal goal is to make true positives close to 0 and true negatives close to infinity. The default is Linear and is used for both training and testing.

2 bbox detected. First bbox has correct answer `none`:

| Algorithm | Distance VS suspect | Distance VS hostage |
|:---------:|:-------------------:|:-------------------:|
|   Cubic   |        1.44         |        1.12         |
|  Linear   |        1.46         |        1.15         |
|  Nearest  |        1.34         |        0.97         |
|   Area    |        1.36         |        1.09         |

Using `Nearest` would have caused a FP on `hostage`, though very close to the threshold. However, `Nearest` clearly outperforms `Cubic` and `Linear` in other examples below.

Second has correct answer `suspect`:

| Algorithm | Distance VS suspect | Distance VS hostage |
|:---------:|:-------------------:|:-------------------:|
|   Cubic   |        1.44         |        1.56         |
|  Linear   |        1.44         |        1.55         |
|  Nearest  |        0.69         |        1.30         |
|   Area    |        0.67         |        1.52         |

In this case, `Cubic` and `Linear` clearly underperformed `Nearest` or `Area`. They produced shockingly different distances for the `suspect` query. `Area` outperformed `Nearest` as the distances are wider apart.

We have done similar tests on samples of the hostage and result are in a uniform trend that clearly shows `Area` > `Nearest` > `Cubic` ~= `Linear`. If you would like more comparison data, you can open a Discussion. We also have compared cases where all 4 algorithms getting it correct, and the performance in terms of distance produced is on par. In other word, there is little downside but huge upside by switching from `Linear` to `Area`.

This may be against many's intuition as `Linear` or `Cubic` are supposed to be of much finer detail than `Nearest` and `Area` when doing scaling, and are often much preferred over the latter 2. However, we think that the large difference between the source resolutions (1000+ px vs 100- px) made uniformity much more important than preserving details. Both `Cubic` and `Linear` can produce images that contains details which may actually negatively impact the model, since many details that are present in the query image may not exist in gallery, or exist in a very different form due to camera blur etc. While fast and less 'fine-grained' algorithms like `Nearest` and `Area` does not include these details that proved to be unnecessary for REID task, thus is able to perform better. This hypothesis can be supported by our leaderboard score produced on the qualifiers test set using different scaling algorithms, where `Linear` scored best and `Nearest` scored about 8mAP lower (which is a massive degradation). This shows that when query and gallery are of similar resolution, using a higher-identity algorithm is preferred, but when they are of very different resolution, using a lower-identity algorithm can be better.

## Speaker Identification
All the code for this task is in [SpeakerID](SpeakerID). It was only introduced to finalists and is not part of qualifiers.

This task proved to be very challenging as:
1. The noise injected into the training and test samples are very strong and overwhelms the speaking voice
2. Team PALMTREE decided to whisper into all their recordings, and when overlaid with the noise, it is very hard to hear anything
3. Very small training and validation dataset: one 15s sample per class for training, and total of five 15s samples for validation, covering only 5/32 classes.
4. Shortage of time to train and experiment

### Data Preprocessing - denoising

### Data Augmentation

### Model

### Training

### Inference

### Finals-specific tuning

## Robot

### Localization

### Path planning

### Movement
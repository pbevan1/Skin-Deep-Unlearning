# Skin Deep Unlearning: Artefact and Instrument Debiasing in the Context of Melanoma Classification

## Method:
 *"Convolutional Neural Networks have demonstrated dermatologist-level performance in the classification of melanoma and
 other skin lesions, but prediction irregularities due to bias are an issue that should be addressed before widespread
 deployment is possible. In this work, we robustly remove bias and spurious variation from an automated melanoma
 classification pipeline using two leading bias 'unlearning' techniques: 'Learning Not to Learn'
[[1]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Learning_Not_to_Learn_Training_Deep_Neural_Networks_With_Biased_CVPR_2019_paper.pdf)
(LNTL) and 'Turning a Blind Eye' [[2]](https://www.robots.ox.ac.uk/~vgg/publications/2018/Alvi18/alvi18.pdf) (TABE),
as well as an additional hybrid of the two (CLGR) . We show that the biases introduced
 by surgical markings and rulers presented in previous studies can be reasonably mitigated using these bias removal
 methods. We also demonstrate the generalisation benefits of 'unlearning' spurious variation relating to the
 imaging instrument used to capture lesion images. The novel contributions of this work include a comparison of
 different debiasing techniques for artefact bias removal and the concept of instrument bias 'unlearning' for
 domain generalisation in melanoma detection. Our experimental results provide evidence that the effect of each of the
 aforementioned biases are notably reduced, with different debiasing techniques excelling at different tasks."*

[[Bevan and Atapour-Abarghouei, 2021](INSERT ARXIV LINK HERE)]
<br>

![Exemplar Results](https://github.com/pbevan1/Skin-Deep-Unlearning/blob/main/images/SM_RU.png)
Examples of artefacts seen in ISIC 2020 data. Top row shows images with surgical markings present, bottom row shows images with rulers present.

![architectures](https://github.com/pbevan1/Skin-Deep-Unlearning/blob/main/images/LNTL_TABE-01.jpg)
'Learning Not to Learn' architecture (left) and 'Turning a Blind Eye' architecture (right). Feature extractor, f, is
implemented as a convolutional architecture such as ResNeXt or EfficientNet in this work. 'fc' denotes a fully connected layer.

---
---
## Usage 

### Software used (see `requirements.txt` for package requirements)

Python 3.9.6

CUDA Version 11.3

Nvidia Driver Version: 465.31

PyTorch 1.8.1

---
### Downloading the data

A free account must be created to download The Interactive Atlas of Dermoscopy, available at this link:
[https://derm.cs.sfu.ca/Download.html](https://derm.cs.sfu.ca/Download.html). Place the `release_v0.zip` file into the
`data/raw_images` directory (see below), from which it will be processed by the `download.py` script. The other datasets
will be automatically downloaded and processed by the `download.py` script.

<pre>
Melanoma-Bias  
└───Data
│   └───csv
│   |   │   asan.csv
│   |   │   atlas.csv
│   |   │   ...
│   |
|   └───images
|   |
|   └───raw_images
|       |   <b>release_v0.zip</b>
|
...
</pre>

Run `download.py` to download, crop and resize the ISIC, ASAN, MClass clinical, MClass dermoscopic and Fitzpatrick17k
datasaets. Have patience as it may take around an hour to complete. The 256x256 resized images are automatically placed
into `data/images` as shown below. The manually downloaded Atlas data (`data/raw_images/release_v0.zip`) will also be
processed by this script. Note this script clears the `data/images` directory before populating it, so if you want to put other
images in there, do this *after* running the `download.py` script.

**NOTE**: The surgical markings/rulers test set from Heidelberg University [[3]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6694463/) is not publicly available.

The data directory should now look as follows:
<pre>
Melanoma-Bias  
└───Data
│   └───csv
│   |   │   asan.csv
│   |   │   atlas.csv
│   |   │   ...
│   |
|   └───images
|       |   asan_256
|       |   atlas_256
|       |   isic_19_train_256
|       |   isic_20_train_256
|       |   MClassC_256
|       |   MClassD_256
|
...
</pre>

If you do wish to manually download the datasets, they are available at the following links:

ISIC 2020 data: [https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256](https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256)  
ISIC 2019/2018/2017 data: [https://www.kaggle.com/cdeotte/jpeg-isic2019-256x256](https://www.kaggle.com/cdeotte/jpeg-isic2019-256x256)  
Interactive Atlas of Dermoscopy: [https://derm.cs.sfu.ca/Welcome.html](https://derm.cs.sfu.ca/Welcome.html)  
ASAN Test set: [https://figshare.com/articles/code/Caffemodel_files_and_Python_Examples/5406223](https://figshare.com/articles/code/Caffemodel_files_and_Python_Examples/5406223)  
MClassC/MClassD: [https://skinclass.de/mclass/](https://skinclass.de/mclass/)

---
### Training and evaluation

Training commands for the main experiments from the paper are below. Please see `arguments.py` for the full range of arguments if you wish to devise alternative experiments. Test results (plots, logs and weights) will autosave into the `results` directory, in subdirectories specific to the test number. Please contact me if you require trained weights for any model in the paper.

Some useful arguments to tweak the below commands:
* Adjust `--CUDA_VISIBLE_DEVICES` and `num-workers` to suit the available GPUs and CPU cores respectively on your machine.
* To run in debug mode add `--DEBUG` (limits epochs to 3 batches).
* To chnage the random seed (default 0), use `--seed` argument.
* To run on different architechtures, use `--arch` argument to choose from `resnext101`, `enet`, `resnet101`, `densenet` or `inception` (default=`resnext101`).
* Add `--cv` to perform cross validation.
* Add `--test-only` if you wish to load weights and run testing only (loads weights of whatever `--test-no` argument is passed).
---
***Unlearning instruments for generalisation:***
<pre>
<b>Baseline:</b> python train.py --test-no 9 --n-epochs 4 --CUDA_VISIBLE_DEVICES 0,1
<b>LNTL:</b> python train.py --test-no 10 --n-epochs 4 --debias-config LNTL --GRL --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8
<b>TABE:</b> python train.py --test-no 11 --n-epochs 4 --debias-config TABE --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8
<b>CLGR:</b> python train.py --test-no 12 --n-epochs 4 --debias-config TABE --GRL --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8
</pre>
---
***Surgical marking bias removal (REQUIRES PRIVATE HEIDELBERG UNIVERSITY DATASET):***
<pre>
<b>Baseline:</b> python train.py --test-no 1 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --marked --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_marked
<b>LNTL:</b> python train.py --test-no 2 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --debias-config LNTL --GRL --marked --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_marked
<b>TABE:</b> python train.py --test-no 3 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --debias-config TABE --marked --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_marked
<b>CLGR:</b> python train.py --test-no 4 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --debias-config TABE --GRL --marked --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_marked
</pre>
---
***Ruler bias removal (REQUIRES PRIVATE HEIDELBERG UNIVERSITY DATASET):***
<pre>
<b>Baseline:</b> python train.py --test-no 5 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --rulers --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_rulers
<b>LNTL:</b> python train.py --test-no 6 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --debias-config LNTL --GRL --rulers --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_rulers
<b>TABE:</b> python train.py --test-no 7 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --debias-config TABE --rulers --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_rulers
<b>CLGR:</b> python train.py --test-no 8 --arch enet --enet-type efficientnet_b3 --n-epochs 15 --debias-config TABE --GRL --rulers --CUDA_VISIBLE_DEVICES 0,1 --skew --heid-test_rulers
</pre>
---
### Double header experiments for generalisation

***ResNeXt-101 double headers (removing instrument and surgical marking bias):***
<pre>
<b>TABE (instrument) + TABE (marks):</b> python train.py --test-no 21 --n-epochs 4 --debias-config doubleTABE --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>CLGR (instrument) + CLGR (marks):</b> python train.py --test-no 22 --n-epochs 4 --debias-config doubleTABE --GRL --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>LNTL (instrument) + CLGR (marks):</b> python train.py --test-no 23 --n-epochs 4 --debias-config both --GRL --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>CLGR (instrument) + LNTL (marks):</b> python train.py --test-no 24 --n-epochs 4 --debias-config both --GRL --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux2 8 --switch-heads --lr-class 0.0003
<b>LNTL (instrument) + TABE (marks):</b> python train.py --test-no 25 --n-epochs 4 --debias-config both --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>TABE (instrument) + LNTL (marks):</b> python train.py --test-no 26 --n-epochs 4 --debias-config both --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux2 8 --switch-heads --lr-class 0.0003
<b>LNTL (instrument) + LNTL (marks):</b> python train.py --test-no 27 --n-epochs 4 --debias-config doubleLNTL --instrument --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
</pre>
---
***ResNeXt-101 double headers (removing instrument and ruler bias):***
<pre>
<b>TABE (instrument) + TABE (rulers):</b> python train.py --test-no 21 --n-epochs 4 --debias-config doubleTABE --instrument --rulers --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>CLGR (instrument) + CLGR (rulers):</b> python train.py --test-no 22 --n-epochs 4 --debias-config doubleTABE --GRL --instrument --rulers --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>LNTL (instrument) + CLGR (rulers):</b> python train.py --test-no 23 --n-epochs 4 --debias-config both --GRL --instrument --rulers --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>CLGR (instrument) + LNTL (rulers):</b> python train.py --test-no 24 --n-epochs 4 --debias-config both --GRL --instrument --rulers --CUDA_VISIBLE_DEVICES 0,1 --num-aux2 8 --switch-heads --lr-class 0.0003
<b>LNTL (instrument) + TABE (rulers):</b> python train.py --test-no 25 --n-epochs 4 --debias-config both --instrument --rulers --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
<b>TABE (instrument) + LNTL (rulers):</b> python train.py --test-no 26 --n-epochs 4 --debias-config both --instrument --rulers --CUDA_VISIBLE_DEVICES 0,1 --num-aux2 8 --switch-heads --lr-class 0.0003
<b>LNTL (instrument) + LNTL (rulers):</b> python train.py --test-no 27 --n-epochs 4 --debias-config doubleLNTL --instrument --rulers --CUDA_VISIBLE_DEVICES 0,1 --num-aux 8 --lr-class 0.0003
</pre>
---
This work is created as part of the project published in the following. The model has been re-trained to provide better quality visual results.
## Reference:

[Skin Deep Unlearning: Artefact and Instrument Debiasing in the Context of Melanoma Classification](INSERT ARXIV LINK HERE)
(P. Bevan, A. Atapour-Abarghouei) [[pdf](INSERT ARXIV LINK HERE)]

```
@InProceedings{,
  author =,
  title =,
  booktitle=,
  pages=,
  year =,
  publisher = 
}

```
---
# Non-Local Recurrent Network for Image Restoration (NeurIPS 2018)

[Paper](http://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf) | [Bibtex](#Bibtex)

WIP: fast evaluation with custom ops

An older version of the NLRN code can be found [here](https://github.com/Ding-Liu/NLRN_v0).

## Usage
### Denoising
#### Preparing BSD500 for training
```
mkdir -p data/bsd500
wget -O data/bsd500/BSR_bsds500.tgz http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
`cd data/bsd500 && tar -xvf BSR_bsds500.tgz`
mkdir -p data/bsd500/flist1
find data/bsd500/BSR/BSDS500/data/images/train/*.jpg data/bsd500/BSR/BSDS500/data/images/test/*.jpg > data/bsd500/flist1/train.flist
find data/bsd500/BSR/BSDS500/data/images/val/*.jpg > data/bsd500/flist1/eval.flist
```
#### Preparing Set12 and BSD68 for evaluation
```
git clone https://github.com/cszn/DnCNN.git data/denoise
find data/denoise/testsets/Set12/*.png > data/set12.flist
find data/denoise/testsets/BSD68/*.png > data/bsd68.flist
```
#### Training on flist1 (train and test) of BSD500
```
python trainer.py --dataset denoise --train-flist data/bsd500/flist1/train.flist --eval-flist data/bsd500/flist1/eval.flist --model nlrn --job-dir debug
# or incremental trainer by number of recurrent states
python incremental_trainer.py --dataset denoise --train-flist data/bsd500/flist1/train.flist --eval-flist data/bsd500/flist1/eval.flist --model nlrn --job-dir debug
```
#### Pre-trained models
12 recurrent states/with correlation propagation: [sigma 15](https://drive.google.com/open?id=1cbaLYjPn_H6TRbLPfA6B3j45TmKdUrfa), [sigma 25](https://drive.google.com/open?id=1l_G9wniOKSM4dS8NqGP-SptPVT5PFo6l), [sigma 50](https://drive.google.com/open?id=1UlDZLSb0a-1fpMgi_LRWozpRsOfWIMek).

15 recurrent states/without correlation propagation: [sigma 15](https://github.com/Ding-Liu/NLRN/files/2674584/sigma15.zip), [sigma 25](https://github.com/Ding-Liu/NLRN/files/2674585/sigma25.zip), [sigma 50](https://github.com/Ding-Liu/NLRN/files/2674586/sigma50.zip).
#### Prediction on Set12 and BSD68
```
python -m datasets.denoise --noise-sigma SIGMA --model-dir MODEL_DIR --input-dir data/denoise/testsets/Set12 --output-dir ./output/Set12
python -m datasets.denoise --noise-sigma SIGMA --model-dir MODEL_DIR --input-dir data/denoise/testsets/BSD68 --output-dir ./output/BSD68
```
`MODEL_DIR` is the directory of `tf.saved_model` and located in `export/Servo/` of `job_dir`.

### Super-resolution
#### Preparing Set5 Set14 BSD100 Urban100 for evaluation
```
wget -O data/SR_testing_datasets.zip http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip
`cd data/ && unzip SR_testing_datasets.zip`
```

## Bibtex
```
@inproceedings{liu2018non,
  title={Non-Local Recurrent Network for Image Restoration},
  author={Liu, Ding and Wen, Bihan and Fan, Yuchen and Loy, Chen Change and Huang, Thomas S},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1680--1689},
  year={2018}
}
```

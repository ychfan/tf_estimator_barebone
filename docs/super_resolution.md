# Super Resolution from a Single Image

## DIV2K dataset
[DIV2K dataset: DIVerse 2K resolution high quality images as used for the NTIRE challenge on super-resolution @ CVPR 2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Download and organize data like: 
```bash
tf_estimator_barebone/data/DIV2K/
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
    └── X2
    └── X3
    └── X4
```

### Dependencies
```bash
conda install tensorflow-gpu pillow
```

### Inference
```bash
python -m dataset.div2k --model-dir MODEL_DIR --input-dir INPUT_DIR --output-dir OUTPUT_DIR
```

### EDSR
Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]

#### Training
```bash
python trainer.py --dataset div2k --model edsr --job-dir ./div2k_edsr
```

### WDSR
Jiahui Yu, Yuchen Fan, Jianchao Yang, Ning Xu, Zhaowen Wang, Xinchao Wang, Thomas Huang, **"Wide Activation for Efficient and Accurate Image Super-Resolution"**, arXiv preprint arXiv:1808.08718. [[arXiv](https://arxiv.org/abs/1808.08718)] [[Code](https://github.com/JiahuiYu/wdsr_ntire2018)]

#### Training
```bash
python trainer.py --dataset div2k --model wdsr --job-dir ./div2k_wdsr
```

### Performance
Compare with [WDSR (PyTorch-based)](https://github.com/JiahuiYu/wdsr_ntire2018#overall-performance)
| Network | Parameters | DIV2K (val) PSNR | Pre-trained models |
| - | - | - | - |
| EDSR Baseline | 1,191,324 | 34.63 | [Download](https://github.com/ychfan/tf_estimator_barebone/files/2502372/edsr.zip)
| WDSR Baseline | 1,190,100 | 34.78 | [Download](https://github.com/ychfan/tf_estimator_barebone/files/2502414/wdsr.zip)
# TF Estimator Barebone
TensorFlow project template with high-level API

## Examples

### CIFAR-10 with ResNet
Usage:
```bash
python trainer.py --dataset cifar10 --model resnet --job-dir ./cifar10
python trainer.py --dataset cifar10 --model resnet --mixup 1.0 --job-dir ./cifar10_mixup
```
Accurarcy: 94.09% without mixup, 94.89% with mixup

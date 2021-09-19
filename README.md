# Paper name(CHAGNE)
This repository is the official PyTorch implementation of "[Paper Name(CHANGE+LINK)]()" by [Author1(change link)]() and [Author2(CHANGE LINK)]().

## Requirements

Tested with the following packages:
- python 3.9
- torch 1.9
- torchvision 0.10
- CUDA 11.1
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) 3.0

## Training

To train the model(s) in the paper, run this command:

```train
python contrastive.py
```

## Evaluation

To evaluate my model linear evaluation and robustness, run:

```eval
python evaluation.py
```

## Results

Our model achieves the following performance on :

### Classification and robustness on CIFAR 10

| Model              |    Accuracy     |   robustness(Under PGD-10)   |
| ------------------ |---------------- | ---------------------------- |
|  ResNet18          |    92.46 %      |            60.47 %           |


## Citation
```

# Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training

This repository is the official PyTorch implementation of "[Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training](https://arxiv.org/abs/2203.08959)" by Adir Rahamim and Itay Naeh.

## Requirements

Tested with the following packages:
- python 3.9
- torch 1.9
- torchvision 0.10
- CUDA 11.1
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) 3.0

## Training and Evaluation

To train and then evaluate the model in the paper, run this command:

```train
python main.py
```

## Results

Our model achieves the following clean accuracy and robustness on the CIFAR-10 dataset:

| Model              |    Accuracy     |   robustness   |
| ------------------ |---------------- | ---------------- |
|  ResNet18          |    92.46 %      |    60.47 %      |


## Citation
```
@inproceedings{Rahamim2022RobustnessTC,
  title={Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training},
  author={Adir Rahamim and Itay Naeh},
  year={2022}
}
```

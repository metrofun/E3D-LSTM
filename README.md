# Eidetic 3D LSTM in PyTorch

This is an unofficial and partial PyTorch implementation of "Eidetic 3D LSTM: A Model for Video Prediction and Beyond" [1]

Implementeds E3D-LSTM and a trainer for traffic flow prediction on TaxiBJ dataset[2]

<img src="./taxibj" width="300" />

## Modifications
* By default uses a cheaper "Scaled Dot-Product"[3] attention.
* Adds more "LayerNorm"[4] for faster training.

## Installation

1. [Download](https://github.com/lucktroy/DeepST/tree/master/data/TaxiBJ) TaxiBJ[4] dataset into `./data/` folder.
2. Install dependencies from `Pipfile`. By default installs CPU-only Pytorch.

## Usage

```bash
python src/trainer.py
```

## Todo
* [ ] Fix TODOs
* [ ] Do qualitative verification.
* [ ] Introduce configs
* [ ] Add visuals. 

## References
[1] Y Wang, L Jiang, MH Yang, LJ Li, M Long, L Fei-Fei. Eidetic 3D LSTM: A Model for Video Prediction and Beyond.  
[2] Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction.  
[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need.  
[4] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization.  

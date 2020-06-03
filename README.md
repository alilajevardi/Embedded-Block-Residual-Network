# Embedded-Block-Residual-Network
EBRN is a recursive restoration model for single-image super-resolution
This repository is a try to implement the EBRN model through Tensorflow and Keras API. As mentioned in the paper, the model was originally implemented on Pythorch, however their original code is not publicly available.

The EBRN was reported in the original paper here (https://ieeexplore.ieee.org/abstract/document/9010860). Some review articles mentioned its power in single-image super-resolution.

The jupyter file of Implementation_v02.1.ipynb is the start point.
To start training, div2k images need to be dwonloded, inzipped and stored in /.div2k/images folder. Currently the scale of 4 (X4) is activated. Thus the following folders need to be present:
/DIV2K_train_HR
/DIV2K_valid_HR
/DIV2K_train_LR_bicubic/X4 
/DIV2K_valid_LR_bicubic/X4

The EBRN model with 4 BRM units created via graphviz and pydot: ![picture](https://github.com/alilajevardi/Embedded-Block-Residual-Network/blob/master/assets/SR_EBRNet_v02.1.png)

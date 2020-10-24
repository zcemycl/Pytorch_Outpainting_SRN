# Pytorch_Outpainting_SRN
This repository rewrites tensorflow implementation of [Wide-Context Semantic Image Extrapolation](http://jiaya.me/papers/imgextrapolation_cvpr19.pdf) paper into PyTorch causally. It refers to the github repository [Jia-Research-Lab/outpainting_srn](https://github.com/Jia-Research-Lab/outpainting_srn), and copies some files `pytorch/{model,util}` from [shepnerd/inpainting_gmcnn](https://github.com/shepnerd/inpainting_gmcnn) to compute `IDMRFLoss` of `VGG19`.
## Progress
This repo aims to train both generator and discriminator from scratch, except the pretrained VGG19 model. It only implements part of the tensorflow one, i.e. subpixel convolution, SegmenticRegerenationNet and relative spatial variant mask. Training the model with `VGG19 IDMRFLoss` fails to converge, so I exclude that loss at the current stage. 

## How to run?
1. Download the `cat2dog` dataset, only use `TrainB` folder for model training. 
2. Run the file to initiate the training progress. 
```python
python main.py [--batch_size][--epoch][--g_cnum][--d_cnum][--gan_loss_alpha]
[--wgan_gp_lambda][--pretrain_l1_alpha][--l1_loss_alpha][--ae_loss_alpha][--fa_alpha]
```
3. Run Tensorboard via,
```
tensorboard --logdir=log/store
```

# HairGANs [Tensorflow]
The Networks for "Hair-GAN: Recovering 3D Hair Structure from a Single Image using Generative Adversarial Networks"

Ref.to [https://arxiv.org/abs/1811.06229]


![Image text](https://github.com/MengZephyr/HairGANs/blob/master/NetworkOverview.png)
An overview of our Hair-GANs architecture. The generator and the discriminator are trained in conjunction.

## ./Ori_Conf_from_Img
This folder contains the code for orientation and confidence map generation with an image as input. Please begin from the "toRunConfidenceBuider" to track the implimentation of [Unsupervised Texture Segmentation Using Gabor
Filters](https://www.ee.columbia.edu/~sfchang/course/dip/handout/jain-texture.pdf)

## Citation
If you use our code or model, please cite our paper:

  @article{zhang2019hair, title={Hair-GAN: Recovering 3D hair structure from a single image using generative adversarial networks}, author={Zhang, Meng and Zheng, Youyi}, journal={Visual Informatics}, volume={3}, number={2}, pages={102–112}, year={2019}, publisher={Elsevier} }

# High-Order-PaddlePaddle
This is a repo for PaddlePaddle implementation of Deep High-Order Pooling Neural Networks, mainly consisting of following works:

CVPR2020 paper ([What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective](https://arxiv.org/abs/2003.11241)([poster](https://github.com/ZhangLi-CS/GCP_Optimization/blob/master/poster.png)))

NeurIPS2021 paper ([Temporal-attentive Covariance Pooling Networks for Video Recognition](https://arxiv.org/abs/2110.14381)([poster](https://github.com/ZilinGao/Temporal-attentive-Covariance-Pooling-Networks-for-Video-Recognition/blob/main/Fig/arch.png)))

CVPR2022 paper ([Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification](https://arxiv.org/pdf/2204.04567.pdf)([poster](https://camo.githubusercontent.com/c4928480aaf790d75df644a1cd22f0f0fd0d2e5037cd9c33f4c2bb006fbf0f12/687474703a2f2f7065696875616c692e6f72672f446565704244432f696c6c757374726174696f6e2e676966)))

NeurIPS2022 paper ([DropCov: A Simple yet Effective Method for Improving Deep Architectures](https://openreview.net/forum?id=QLGuUwDx4S) ([poster](https://github.com/mingzeG/DropCov/blob/main/figures/overview.jpg)))

# Install
To use the model in this repo, you should clone the repo and install paddlepaddle-gpu:
```
git clone https://github.com/mingzeG/High-Order-PaddlePaddle.git
conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge 
```

# Citation
```
@inproceedings{wang2020deep,
  title={What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective},
  author={Wang, Qilong and Zhang, Li and Wu, Banggu and Ren, Dongwei and Li, Peihua and Zuo, Wangmeng and Hu, Qinghua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

```
@article{gao2021temporal,
  title={Temporal-attentive Covariance Pooling Networks for Video Recognition},
  author={Gao, Zilin and Wang, Qilong and Zhang, Bingbing and Hu, Qinghua and Li, Peihua},
  journal={NeurIPS},
  year={2021}
}
```

```
@inproceedings{DeepBDC-CVPR2022,
    title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
    author={Jiangtao Xie and Fei Long and Jiaming Lv and Qilong Wang and Peihua Li}, 
    booktitle={CVPR},
    year={2022}
 }
```

```
@inproceedings{wang2022nips,
  title={A Simple yet Effective Method for Improving Deep Architectures},
  author={Qilong Wang and Mingze Gao and Zhaolin Zhang and Jiangtao Xie and Peihua Li and Qinghua Hu},
  booktitle = {NeurIPS},
  year={2022}
}
```

# Acknowledgement
Our code are built following 
[GCP_Optimization](https://github.com/ZhangLi-CS/GCP_Optimization),
[DeepBDC](https://github.com/Fei-Long121/DeepBDC),
[TCPNet](https://github.com/ZilinGao/Temporal-attentive-Covariance-Pooling-Networks-for-Video-Recognition),
[DropCov](https://github.com/mingzeG/DropCov)
, thanks for their excellent work

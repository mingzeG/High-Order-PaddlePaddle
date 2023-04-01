# DeepBDC for few-shot learning
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/DeepBDC/illustration.gif" width="80%"/>
</div>


## Introduction
In this repo, we provide the implementation of the following paper:<br>
"Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification"   [[Project]](http://peihuali.org/DeepBDC/index.html) [[Paper]](https://arxiv.org/pdf/2204.04567.pdf).

 In this paper, we propose deep Brownian Distance Covariance (DeepBDC) for few-shot classification. DeepBDC can effectively learn image representations by measuring, for the query and support images, the discrepancy between the joint distribution of their embedded features and product of the marginals. The core of DeepBDC is formulated as a modular and efficient layer, which can be flexibly inserted into deep networks, suitable not only for meta-learning framework based on episodic training, but also for the simple transfer learning (STL) framework of pretraining plus linear classifier.<br>

 If you find this repo helpful for your research, please consider citing our paper：<br>
```
@inproceedings{DeepBDC-CVPR2022,
    title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
    author={Jiangtao Xie and Fei Long and Jiaming Lv and Qilong Wang and Peihua Li}, 
    booktitle={CVPR},
    year={2022}
 }
```
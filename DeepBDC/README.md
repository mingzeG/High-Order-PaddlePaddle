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

## Training usage
In this repository we just provided the code of TCPNet framework. If you want to train or eval our method, pleade follow the usage of  [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo), a toolset for video tasks prepared for the industry and academia provided by PaddlePaddle Official.
For example:

```
DATA_ROOT=/path/mini_imagenet

python pretrain.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu 0 --lr 5e-2 --t_lr 1e-3 --epoch 170 --milestones 100 150 --save_freq 100 --reduce_dim 640 --dropout_rate 0.8 --val meta --val_n_episode 600

```
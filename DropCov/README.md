# DropCov

## Introduction
Post-normalization plays a key role in deep global covariance pooling (GCP) networks. In this paper, we for the first time show that effective post-normalization can make a good trade-off between representation decorrelation and information preservation for GCP, which are crucial to alleviate over-fitting and increase representation ability of deep GCP networks, respectively. Based on this finding, we propose a simple yet effective pre-normalization method for GCP (namely DropCov), which performs an adaptive channel dropout before GCP to achieve tradeoff between representation decorrelation and information preservation. The proposed DropCov improves performance of both deep CNNs and ViTs.
![Poster](figures/overview.jpg)

## Pretrained models
<table>
  <tr>
    <th>arch</th>
    <th>top1</th>
    <!-- <th>top5</th> -->
    <!-- <th>log</th> -->
    <th colspan="6">checkpoint</th>
  </tr>
  <tr>
    <td>ResNet18_ACD</td>
    <td>73.36%</td>
    <td><a href="https://drive.google.com/file/d/1e-eeZUa40BUMiSN5ASQ5Z37k0Lh54DE7/view?usp=sharing">full ckpt</a></td>
  </tr>
  <tr>
    <td>ResNet50_ACD</td>
    <td>75.84%</td>
    <td><a href="https://drive.google.com/file/d/1e-eeZUa40BUMiSN5ASQ5Z37k0Lh54DE7/view?usp=sharing">full ckpt</a></td>
  </tr>
<table>

## Training usage
In this repository we just provided the code of Dropcov method and classical CNN architectures (ResNet) that use this method. If you want to train or eval our method, pleade follow the usage of  [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), an image classification and image recognition toolset provided by PaddlePaddle Official.
For example:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./src/config/ResNet50_acd.yaml
```
## Config setting
We also provide the parameter files for training, which are later needed in paddleclas. You can find in thd directory of src/config. By adding our model and config file to PaddleClas, you can easily reproduce our result.

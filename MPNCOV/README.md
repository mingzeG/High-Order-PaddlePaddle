# Matrix Power Normalization
This is a PaddlePaddle implementation of CVPR2020 paper ([What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective](https://arxiv.org/abs/2003.11241)([poster](https://github.com/ZhangLi-CS/GCP_Optimization/blob/master/poster.png))). 

## Pretrained models
<table>
  <tr>
    <th>arch</th>
    <th>top1</th>
    <th>top5</th>
    <th>log</th>
    <th colspan="6">checkpoint</th>
  </tr>
  <tr>
    <td>ResNet50_MPNCOV</td>
    <td>78.55%</td>
    <td>94.21%</td>
    <td><a href="https://drive.google.com/file/d/17yKzvJyGBgzRMDvW-KiQbF2G_M2uDnn3/view?usp=share_link">log</a></td>
    <td><a href="https://drive.google.com/file/d/1Owpw38UlOjHp1IPz9QfGynbeWXJywOsI/view?usp=share_link">full ckpt</a></td>
  </tr>
<table>

## Training usage
In this repository we just provided the code of MPNCOV method and some classical CNN architectures (ResNet, MobileNetV2, ShuffleNetV2) that use this method. If you want to train or eval our method, pleade follow the usage of  [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), an image classification and image recognition toolset provided by PaddlePaddle Official.

## Config setting
We also provide the parameter files for training, which are later needed in paddleclas. You can find in thd directory of src/config. By adding our model and config file to PaddleClas, you can easily reproduce our result.
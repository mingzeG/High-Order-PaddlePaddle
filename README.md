# PaddlePaddle MPNCOV
This is a repository for MPNCOV(Matrix Power Normalization) method, which is implemented in PaddlePaddle.

## Pretrained models
Here we provide the pretrained checkpoint and train log of ResNet50_MPNCOV. You can choose to check the train log or download the weights of the pretrained backbone to reproduce our result. 

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
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">log</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth">full ckpt</a></td>
  </tr>
<table>

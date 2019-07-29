# MAH-Pytorch
PyTorch implementation of our paper "Collaborative Learning for Extremely Low Bit Asymmetric Hashing" [[Link]](https://arxiv.org/abs/1809.09329). 

## Preparation
### Dependencies
- Python 2.7
- PyTorch (version >= 0.4.1)


### Datasets
- CIFAR download the CIFAR-10 Matlab version [[Link]](https://www.cs.toronto.edu/~kriz/cifar.html) then run the script
```shell
matlab ./data/CIFAR-10/SaveFig.m
```
- NUSWIDE dataset
- MIRFlickr dataset
([referenced repo](https://github.com/jiangqy/DPSH-pytorch))


## Results

### Mean Average Precision on CIFAR-10.
<table>
    <tr>
        <td rowspan="2">Net Structure</td><td rowspan="2">PlatForm</td>    
        <td colspan="4">Code Length</td>
    </tr>
    <tr>
        <td >12 bits</td><td >24 bits</td> <td >32 bits</td><td >48 bits</td>  
    </tr>
    <tr>
        <td >VGG-F</td><td >MatConvNet</td ><td > 0.713 </td> <td > 0.727 </td><td > 0.744</td><td > 0.757</td>  
    </tr>
    <tr>
        <td >Alexnet</td><td >Pytorch</td ><td > 0.7505</td> <td > 0.7724 </td><td > 0.7758 </td> <td > 0.7828 </td>
    </tr>
    <tr>
        <td >VGG-11</td><td >Pytorch</td ><td > 0.7655 </td> <td > 0.8042 </td><td > 0.8070 </td> <td > 0.8108 </td>
    </tr>
</table>


### 
![](./fig/CIFAR.png =200x300)
![](./fig/CIFAR.png =200x300)
![](./fig/CIFAR.png =200x300)
## Usage
For traning with the cascaded multihead structure on different datasets:
```shell
python cascade_CIFAR-10.py --bits '4' --gpu '1' --batch-size 64
python cascade_FLICKR.py --bits '4' --gpu '1' --batch-size 64
python cascade_NUS_WIDE.py --bits '4' --gpu '1' --batch-size 64
```

For traning with the flat multihead structure on different datasets:
```shell
python flat_CIFAR-10.py --bits '4' --gpu '1' --batch-size 64
python flat_FLICKR.py --bits '4' --gpu '1' --batch-size 64
python flat_NUS_WIDE.py --bits '4' --gpu '1' --batch-size 64
```
## Citation
Please cite the following paper in your publications if it helps your research:
    
    @article{DBLP:journals/corr/abs-1809-09329,
        author    = {Yadan Luo and
                     Yang Li and
                     Fumin Shen and
                     Yang Yang and
                     Peng Cui and
                     Zi Huang},
        title     = {Collaborative Learning for Extremely Low Bit Asymmetric Hashing},
        journal   = {CoRR},
        volume    = {abs/1809.09329},
        year      = {2018},
        url       = {http://arxiv.org/abs/1809.09329},
        archivePrefix = {arXiv},
        eprint    = {1809.09329},
        timestamp = {Wed, 13 Mar 2019 15:40:02 +0100},
        biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1809-09329},
        bibsource = {dblp computer science bibliography, https://dblp.org}
      }

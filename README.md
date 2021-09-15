## DeepLabV3

[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) - DeepLabV3

Implementation of **DeepLabV3** using PyTorch

### Available architectures


| DeepLabV3               | Backbone          |
| ----------------------- | ----------------- |
| deeplabv3_resnet50      | resnet50          |
| deeplabv3_resnet101     | resnet101         |
| deeplabv3_mobilenetv3   | mobilenetv3_large |

### Dataset
```
├── COCO 
    ├── annotations
        ├──instances_train2014.json
        ├──instances_val2014.json
    ├── train2014
        ├── COCO_train2014_00000000000.png
        ├── COCO_train2014_00000000001.png
    ├── val2014
        ├── COCO_val2014_00000000000.png
        ├── COCO_val2014_00000000001.png
```

### Train
**Note**

Modify these arguments according to your data and model in `train.py`
```
parser.add_argument('--data-path', default='../../Datasets/COCO', help='dataset path') 
parser.add_argument('--dataset', default='coco', help='dataset name')                  
parser.add_argument('--model', default='deeplabv3_resnet50', help='model')             
```

**Distributed Data Parallel:** 

1. deeplabv3_resnet50:

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.02 --dataset coco -b 8 --model deeplabv3_resnet50

```

1. deeplabv3_resnet101:


```
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.02 --dataset coco -b 8 --model deeplabv3_resnet101

```

1. deeplabv3_mobilenet_v3_large:

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.02 --dataset coco -b 8 --model deeplabv3_mobilenet_v3_large

```

**Without Distributed Data Parallel:** 
```
python train.py
```


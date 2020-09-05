#### This is a fork of https://github.com/mzolfaghari/ECO-pytorch's PaddlePaddle implementation for the [paper](https://arxiv.org/pdf/1804.09066.pdf):
##### " ECO: Efficient Convolutional Network for Online Video Understanding, European Conference on Computer Vision (ECCV), 2018." By Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox
 
 


##### NOTE

* This only test ECO-Full model for dataset UCF101 
* You can try it on AIStudio project https://aistudio.baidu.com/aistudio/projectdetail/698987


### Environment:
* Python 3.7
* PaddlePaddle 1.8.0

### Clone this repo

```
git clone https://github.com/Nullius-2020/ECO-pp
```

### Generate dataset lists

```bash
python gen_dataset_lists.py <ucf101> <dataset_frames_root_path>
```
e.g. python gen_dataset_lists.py ucf101 ~/dataset/ucf101/

> The dataset should be organized as:<br>
> <dataset_frames_root_path>/<video_name>/<frame_images>

### Training

For finetuning on UCF101 use the following command:

    sh run_demo_ECO_Full.sh local
```

### NOTE
* If you want to train your model from scratch change the config as following:
```bash
    --pretrained_parts scratch
```
* configurations explained in "opts.py"

### Thanks

* This project originated from the course https://aistudio.baidu.com/aistudio/education/group/info/1340
   
### Citation
If you use this code or ideas from the paper for your research, please cite this paper:
```
@inproceedings{ECO_eccv18,
author={Mohammadreza Zolfaghari and
               Kamaljeet Singh and
               Thomas Brox},
title={{ECO:} Efficient Convolutional Network for Online Video Understanding},	       
booktitle={ECCV},
year={2018}
}
```


# Adaptation of Deep High-Resolution Representation Learning for Satellite Pose Estimation
## News
- [2024/11/28] In the frame of a project at ESA we create this repository to serve as a reference for future HRNet for satellite pose estimation applications.

## Introduction
This is an adaptation of the official repository of HRNet. For the [original repository](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and its [paper](https://arxiv.org/abs/1902.09212) see here. 
In this work, the original authors were interested in human pose estimation and produced state-of-the-art results for this task. They relied on high-to-low resolution network architecture. This has been succesfully employed by the winning team of the 2019 Kelvins Pose Estimation Challenge of the ESA Advanced Concepts Team and the Stanford Space Rendezvous Laboratory (here their paper: [Satellite Pose Estimation with Deep Landmark Regressionand Nonlinear Pose Reﬁnement](https://arxiv.org/abs/1908.11542)).

The goal of this repository is also to offer a general reference for others who want to adapt HRNet to the space use-case. Below we describe how you are able to add your own dataset and how to use our previously created dataset loading scripts and configuration files to easily run them with your data.

For our datasets we followed the mpii format and the tutorial we provide for your own datasets follows mpii as well.

![Illustrating the result for ENVISAT dataset](/figures/validationPredictionsEnvisat.jpg)
## Main Results
### Results on AIRBUS MAN DATA L2
| Arch               | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|------|----------|-------|-------|------|------|-------|------|----------|
| pose_resnet_50     | 96.4 |     95.3 |  89.0 |  83.2 | 88.4 | 84.0 |  79.6 | 88.5 |     34.0 |
| pose_resnet_101    | 96.9 |     95.9 |  89.5 |  84.4 | 88.4 | 84.5 |  80.7 | 89.1 |     34.0 |
| pose_resnet_152    | 97.0 |     95.9 |  90.0 |  85.0 | 89.2 | 85.3 |  81.3 | 89.6 |     35.0 |
| **pose_hrnet_w32** | 97.1 |     95.9 |  90.3 |  86.4 | 89.1 | 87.1 |  83.3 | 90.3 |     37.7 |

### Note:
- Flip test is used.
- Input size is 256x256
- pose_resnet_[50,101,152] is our previous work of [*Simple Baselines for Human Pose Estimation and Tracking*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html)


## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. We ran this using Google Colab Pro with the A100 runtime.

## Quick start
We strongly suggest using our code with Google Colab. We also provide general guidelines for the use with your own hardware but be aware that this has *not been tested* by us beforehand. 
### Installation (Google Colab)
1. Clone this repository in your Google Drive and mount it in your Google Colab instance. For reference see [here](https://stackoverflow.com/questions/67553747/how-do-i-link-a-github-repository-to-a-google-colab-notebook).
2. Download pretrained models from here [GoogleDrive](https://drive.google.com/drive/folders/1ePWzYcP4PIf772dgVxAQqNXr9-cATCXX?usp=drive_link). Only need the mpii models will be needed for training.
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            `-- pose_mpii
                |-- pose_hrnet_w32_256x256.pth
                |-- pose_hrnet_w48_256x256.pth
                |-- pose_resnet_101_256x256.pth
                |-- pose_resnet_152_256x256.pth
                `-- pose_resnet_50_256x256.pth

   ``` 
3. Data preparation
**For your own data**, you need an image folder, a set of .json files with annotations for training, testing and validation and a Ground Truth file for evaluation. A set of Matlab scripts for creating these can be found in the VISY-REVE Tool: [TBD](https://TBD.com)
**For AIRBUS ESA data**, please download from [TBD](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```
4. In the experiments/satellite folder create your own config yaml file. The values to be changed are annotated in the template.yaml file.
5. Follow the steps laid out in the HRNet_setup notebook file we provide. It takes care of the necessary bugfixes that we encountered in adapting the original HRNet code for our purpose.
6. Enjoy! For training and visualization see below. You can also consult the HRNet_setup file.

### Installation (Own Hardware)
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```



### Training and Testing

#### Testing on MPII dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```



### Inference

#### Inference on existing datasets

TODO
```
python visualization/plot_coco.py \
    --prediction output/coco/w48_384x288_adam_lr1e-3/results/keypoints_val2017_results_0.json \
    --save-path visualization/results

```


[//]: # "Comment"<img src="figures\visualization\coco\score_610_id_2685_000000002685.png" height="215"><img #src="figures\visualization\coco\score_710_id_153229_000000153229.png" height="215"><img #src="figures\visualization\coco\score_755_id_343561_000000343561.png" height="215">

![Illustrating the ground truth](/figures/groundTruthEnvisat.png)


### Other applications
Many other dense prediction tasks, such as segmentation, face alignment and object detection, etc. have been benefited by HRNet. More information can be found at [High-Resolution Networks](https://github.com/HRNet).


### Source
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
```

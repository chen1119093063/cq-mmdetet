# Low-Light Object Detection (EXDark dataset)

**Step 1: Dataset Download**

(1). Download **EXDark** (include images enhancement by MBLLEN, Zero-DCE, KIND) in VOC format from [google drive](https://drive.google.com/file/d/1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC/view?usp=sharing) or [baiduyun](https://pan.baidu.com/s/1m4BMVqClhMks4S0xulkCcA), passwd:1234. For linux system download, directly run: 

```
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC" -O EXDark.tar.gz && rm -rf /tmp/cookies.txt
```

(2). Then unzip:

```
$ tar -zxvf EXDark.tar.gz
```

We have already split the EXDark dataset with train set (80%) and test set (20%), see paper [MAET (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf) for more details.

The EXDark dataset format should be look like:

```
EXDark
│      
│
└───JPEGImages
│   │───IMGS (original low light)
│   │───IMGS_Kind (imgs enhancement by [Kind, mm 2019])
│   │───IMGS_ZeroDCE (imgs enhancement by [ZeroDCE, cvpr 2020])
│   │───IMGS_MEBBLN (imgs enhancement by [MEBBLN, bmvc 2018])
│───Annotations   
│───main
│───label
```

(3). Then change [line1](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/a0e4de1029eab1e6030f11cebbb7aaec2a64360b/IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_detr.py#L3) (IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_detr.py) and [line2](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/a0e4de1029eab1e6030f11cebbb7aaec2a64360b/IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_yolo.py#L2) (IAT_high/IAT_mmdetection/configs/_base_/datasets/exdark_yolo.py) to your own data path.


**Step 2: Enviroment Setting**

Download mmcv 2.0, and download adapte to your own cuda version and torch version:
```
$ conda create --name openmmlab python=3.8 -y
$ conda activate openmmlab
```
Install PyTorch following official instructions, e.g.
On GPU platforms:
```
$ pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
On CPU platforms:
```
$ conda install pytorch torchvision cpuonly -c pytorch
```

Install  mmcv
```
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
```
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
验证
```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
**Step 3: Testing with pretrain model**



**Step 4: Training your own model**



**Others:**

*Baselines on EXDark dataset (renew) on YOLO-V3 object detector:*

| class | Bicycle | Boat | Bottle | Bus | Car | Cat | Chair | Cup | Dog | Motorbike | People | Table | Total |
|  ---- | ----    | ---- | ----   | ----| ----| ----| ----  | ----| ----| ----      | ----   |  ---- |  ---- |
| Baseline |79.8 | 75.3 | 78.1 | 92.3 | 83.0 | 68.0 | 69.0 | 79.0 | 78.0 | 77.3 | 81.5 | 55.5 | 76.4 |
| KIND (MM 2019) | 80.1 | 77.7 | 77.2 | 93.8 | 83.9 | 66.9 | 68.7 | 77.4 | 79.3 | 75.3 | 80.9 | 53.8 | 76.3 |
| MBLLEN (BMVC 2018) | 82.0 | 77.3 | 76.5 | 91.3 | 84.0 | 67.6 | 69.1 | 77.6 | 80.4 | 75.6 | 81.9 | 58.6 | 76.8 |
| Zero-DCE (CVPR 2020) | 84.1 | 77.6 | 78.3 | 93.1 | 83.7 | 70.3 | 69.8 | 77.6 | 77.4 | 76.3 | 81.0 | 53.6 | 76.9 |
| [MAET (ICCV 2021)](https://github.com/cuiziteng/ICCV_MAET) | 83.1| 78.5| 75.6| 92.9| 83.1| 73.4| 71.3| 79.0| 79.8| 77.2| 81.1| 57.0| 77.7|
| [DENet (ACCV 2022)](https://openaccess.thecvf.com/content/ACCV2022/html/Qin_DENet_Detection-driven_Enhancement_Network_for_Object_Detection_under_Adverse_Weather_ACCV_2022_paper.html) | 80.4 | 79.7 | 77.9 | 91.2 | 82.7 | 72.8 | 69.9 | 80.1 | 77.2 | 76.7 | 82.0 | 57.2 | 77.3|
| IAT-YOLO (BMVC 2022) | 79.8 | 76.9 | 78.6 | 92.5 | 83.8 | 73.6 | 72.4 | 78.6 | 79.0 | 79.0 | 81.1 | 57.7 | 77.8 |
| [PEYOLO (ICANN 2023)](https://arxiv.org/pdf/2307.10953.pdf) | 84.7 | 79.2 | 79.3 | 92.5 | 83.9 | 71.5 | 71.7 | 79.7 | 79.7 | 77.3 | 81.8 | 55.3 | 78.0 |

Dataset Citation:

```
@article{EXDark,
  title={Getting to know low-light images with the exclusively dark dataset},
  author={Loh, Yuen Peng and Chan, Chee Seng},
  journal={Computer Vision and Image Understanding},
  year={2019},
}
```

Code Usage Citation:

```
@InProceedings{,
    author    = {Cq},
    title     = {},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
```

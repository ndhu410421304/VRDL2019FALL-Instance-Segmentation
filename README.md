# VRDL2019FALL Instance Segmentation
This is an assignment of implement instance segementation on tiny pascal voc dataset with 1349 labeled image and 100 tested image.
I had implement the task base on two backbone modules:
1. Mask RCNN, from https://github.com/matterport/Mask_RCNN
2. Detectron2(windows build), from https://github.com/conansherry/detectron2

However since the first one is not I had not proper fine-tuned and it did not perform well yet, I would like to only focus on dectron2.
The code I had written were 
1. detectron_train.py https://github.com/ndhu410421304/VRDL2019FALL-Instance-Segmentation/blob/master/Detectron2/detectron_train_mod.py
and 
2. detectron_test_mod_mult.py https://github.com/ndhu410421304/VRDL2019FALL-Instance-Segmentation/blob/master/Detectron2/detectron_test_mod_mult.py

Both of these file were start modify from detectron2's colab notebook https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

## Hardware
The following specs were used to create the original solution.
- Windows10
- Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
- 1x NVIDIA GTX1080 

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Train models](#train-models)
4. [Make Submission](#make-submission)

## Installation
Using Anaconda is strongly recommended.
```
conda create -n torchdet python=3.6
activate torchdet
```
#### Note: following part from detectron2's repository ####

### Requirements
- Python >= 3.6(Conda)
- PyTorch 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install git+https://github.com/facebookresearch/fvcore`
- pycocotools: `pip install cython; pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`
- VS2019(no test in older version)/CUDA10.1(no test in older version)
* Self Note: The installation of pycocotools may be influence by some library, so the installation order may be importance. Make sure to open a new environment for minimum the risk of crash the environment. 

### several files must be changed by manually.
```
file1: 
  {your evn path}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h(190)
    static constexpr size_t DEPTH_LIMIT = 128;
      change to -->
    static const size_t DEPTH_LIMIT = 128;
file2: 
  {your evn path}\Lib\site-packages\torch\include\pybind11\cast.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\pybind11\cast.h(1449)
    explicit operator type&() { return *(this->value); }
      change to -->
    explicit operator type&() { return *((type*)this->value); }
```

### Build detectron2

After having the above dependencies, run:
```
conda activate {your env}

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

git clone https://github.com/conansherry/detectron2

cd detectron2

python setup.py build develop
```
Note: you may need to rebuild detectron2 after reinstalling a different build of PyTorch.

#### Note: above part from detectron2's repository ####

## Dataset Preparation
* You can view my detectron2 directory to see how to locate file correctly. Can ignore the file not mention in following.
### Download files
From https://github.com/NCTU-VRDL/CS_IOC5008/tree/master/HW4 download pascal_train.json, test.json and utils.py
You may need to put the annotation file in root directory:
```
+- pascal_train.json
+- test.json
```
Same for utils.py, which may use the function binarytorle to change binary mask to rle codes:
```
+- utils.py
```
From https://github.com/ndhu410421304/VRDL2019FALL-Instance-Segmentation/tree/master/Detectron2 download detectron_train_mod.py and detectron_test_mod_mult.py, and located them in root directory
```
+- detectron_train_mod.py
+- detectron_test_mod_mult.py
```
No further modification operation need for these five files, all located in root diurectory of detectron 2.
### Prepare Images
Download images from here: https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK
After downloading images, the data directory is structured as:
```
train
  +- xxx.png
  +- xxx2.png
  +- ...
val
  +- xxx.png
  +- xxx2.png
  +- ...
```
This two folder were all training data. Keep the content of them the same.
For testing data, it should be like:
```
test_images
  +- xxx.png
  +- xxx2.png
  +- ...
 ```
For these two contents, no further need for modications excepet duplicate and rename, all located in root diurectory of detectron 2.

## Train models
To train models, run following commands.
```
python detectron_train_mod.py
```
Preview images may pop up if noting goes wrong. Click the image window and press enter until the program start running.
After training, prediction on validation data may pop up as previous mention. Click the window and enter for next image until previews end.

## Make Submission
Following command will ensemble of all models and make submissions.
```
python detectron_test_mod_mult.py
```
* Notice that if you have modify anything in detectron_train_mod.py, change the same setting in detectron_test_mod_mult.py before running.

## File explanation

detectron_train_mod.py is the file we use for training. In this file, you are going to 
1. (29) Implement a get_dict class as data loader, which you may need to record the image path and annotation for each file.
2. (60) Generate catalog files, one for dataset, one for metafile(class name)
3. (77) Setup configurarion.
* If you do not set it up, it will loadtheconfiguration from the backbone you set.
4. (98) Start training!
5. (110) Load the final weight generated, then do validation visualization.
* Here we use same image file in train and validation set for convinience

detectron_test_mod_mult.py is the file we use for generating multiple submission files by check point at a time. In this file, you are going to
1. Same step as step 1-3 as above, expect that this time we only need to load testing data.
* Becuase we do not have annotation file for test data, we just send in toy information, since we are not going to use it; However if you want to do validation, you can follow thwe guide in train part of how to use it.
2. (95) Start testing in loop! Each loop would load onecheckpoint and generate one output.

## Modification guide

Since you may want to customize the training and testing, heres some quick notes:
* (26) You can load your annotatiobn bythisfunction
* (46) May be crucial to understand your own annotation's format
* (49) If your category start from 0, remove -1 (alsochange in test)
* (62) Set up class name in thing_classes corresponmd to id
* (80) Load the config file you like. Config files can be found at config directory.
* (85) Load the weight. Can found the pretrain weight from the config file you select, or change it to the located you put if you had prepared by yourself.
* (92) Need to modify your class count
* Same guidline can be apply to test process, but remember to syncronize them if you are going to do them in a sequence.

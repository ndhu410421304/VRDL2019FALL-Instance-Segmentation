# VRDL2019FALL Instance Segmentation
This is an assignment of implement instance segementation on tiny pascal voc dataset with 1349 labeled image and 100 tested image.
I had implement the task base on two backbone modules:
1. Mask RCNN, from https://github.com/matterport/Mask_RCNN
2. Detectron2(windows build), from https://github.com/conansherry/detectron2
However since the first one is not I had not proper fine-tuned and it did not perform well yet, I would like to only focus on dectron2.

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
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hpa python=3.6
source activate hpa
pip install -r requirements.txt
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
You may need to put the annotation file in root directory:
```
+- pascal_train.json
+- test.json
```
No further modification operation need for these two files
### Prepare Images
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
test_images
  +- xxx.png
  +- xxx2.png
  +- ...
For these two contents, no further need for modications excepet duplicate and rename.

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
Notice that if you have modify anything in detectron_train_mod.py, change the same setting in this file before running

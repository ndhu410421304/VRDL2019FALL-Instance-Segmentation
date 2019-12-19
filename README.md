# VRDL2019FALL Instance Segmentation
This is an assignment of implement instance segementation on tiny pascal voc dataset with 1349 labeled image and 100 tested image.
I had implement the task base on two backbone modules:
1. Mask RCNN, from https://github.com/matterport/Mask_RCNN
2. Detectron2(windows build), from https://github.com/conansherry/detectron2
However since the first one is not I had not proper fine-tuned and it did not perform well yet, I would like to only focus on dectron2.

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
- 1x NVIDIA GTX1080 

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Download Official Image](#download-official-image)
3. [Make RGBY Images](#make-rgby-images) for official.
4. [Download Pretrained models](#pretrained-models)
5. [Inference](#inference)
6. [Make Submission](#make-submission)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hpa python=3.6
source activate hpa
pip install -r requirements.txt
```

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

<div align="center">
  <img src="docs/windows_build.png"/>
</div>

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

## Dataset Preparation
All required files except images are already in data directory.
If you generate CSV files (duplicate image list, split, leak.. ), original files are overwritten. The contents will be changed, but It's not a problem.

### Prepare Images
After downloading and converting images, the data directory is structured as:
```
data
  +- raw
  |  +- train
  |  +- test
  |  +- external
  +- rgby
  |  +- train
  |  +- test
  |  +- external
```
#### Download Official Image
Download and extract *train.zip* and *test.zip* to *data/raw* directory.
If the Kaggle API is installed, run following command.
```
$ kaggle competitions download -c human-protein-atlas-image-classification -f train.zip
$ kaggle competitions download -c human-protein-atlas-image-classification -f test.zip
$ mkdir -p data/raw
$ unzip train.zip -d data/raw/train
$ unzip test.zip -d data/raw/test
```

#### Download External Images
To download external images, run following command. The external images will be located in *data/raw/external*
```
$ python tools/download.py
```

#### Make RGBY Images
To train or inference, converting to RGBY image is required. Run following commands.

For official:
```
$ python tools/make_rgby.py --input_dir=data/raw/train --output_dir=data/rgby/train
$ python tools/make_rgby.py --input_dir=data/raw/test --output_dir=data/rgby/test
```
For external:
```
$ python tools/make_rgby.py --input_dir=data/raw/external --output_dir=data/rgby/external
```

### Generate CSV files
*You can skip this step. All CSV files are prepared in data directory.*

#### Duplicated Image List
There are duplicated images. To search them, run following commands. *duplicates.ahash.csv* and *duplicates.phash.csv* will be generated.
```
$ python tools/find_duplicate_images.py
```

#### Split Dataset
Create 5 folds CV set. One for training, the other for searching augmentation. *split.stratified.[0-4].csv* and *split.stratified.small.[0-4].csv* will be generated.
```
$ python stratified_split.py
$ python stratified_split.py --use_external=0
```

#### Search Data Leak
To learn more about data leak, please, refer to [this post](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534). Following comand will create *data_leak.ahash.csv* and *data_leak.phash.csv*. [The other leak](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73395y) is already in *data* directory.
```
$ python find_data_leak.py
```

## Training
In configs directory, you can find configurations I used train my final models. My final submission is ensemble of resnet34 x 5, inception-v3 and se-resnext50, but ensemble of inception-v3 and se-resnext50's performance is better.

### Search augmentation
To find suitable augmentation, 256x256 image and resnet18 are used.
It takes about 2 days on TitanX. The result(best_policy.data) will be located in *results/search* directory.
The policy that I used is located in *data* directory.
```
$ python train.py --config=configs/search.yml
```

### Train models
To train models, run following commands.
```
$ python train.py --config={config_path}
```
To train all models, run `sh train.sh`

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet34 | 1x TitanX | 512 | 40 | 16 hours
inception-v3 | 3x TitanX | 1024 | 27 | 1day 15 hours
se-resnext50 | 2x TitanX | 1024 | 22 | 2days 15 hours

### Average weights
To average weights, run following commands.
```
$ python swa.py --config={config_path}
```
To average weights of all models, simply run `sh swa.sh`
The averages weights will be located in *results/{train_dir}/checkpoint*.

### Pretrained models
You can download pretrained model that used for my submission from [link](https://www.dropbox.com/s/qo65gw8kml5hgag/results.tar.gz?dl=0). Or run following command.
```
$ wget https://www.dropbox.com/s/qo65gw8kml5hgag/results.tar.gz
$ tar xzvf results.tar.gz
```
Unzip them into results then you can see following structure:
```
results
  +- resnet34.0.policy
  |  +- checkpoint
  +- resnet34.1.policy
  |  +- checkpoint
  +- resnet34.2.policy
  |  +- checkpoint
  +- resnet34.3.policy
  |  +- checkpoint
  +- resnet34.4.policy
  |  +- checkpoint
  +- inceptionv3.attention.policy.per_image_norm.1024
  |  +- checkpoint
  +- se_resnext50.attention.policy.per_image_norm.1024
  |  +- checkpoint
```

## Inference
If trained weights are prepared, you can create files that contains class probabilities of images.
```
$ python inference.py \
  --config={config_filepath} \
  --num_tta={number_of_tta_images, 4 or 8} \
  --output={output_filepath} \
  --split={test or test_val}
```
To make submission, you must inference test and test_val splits. For example:
```
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test_val.csv --split=test_val
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test.csv --split=test
```
To inference all models, simply run `sh inference.sh`

## Make Submission
Following command will ensemble of all models and make submissions.
```
$ python make_submission.py
```
If you don't want to use, modify *make_submission.py*.
For example, if you want to use inception-v3 and se-resnext50 then modify *test_val_filenames, test_filenames and weights* in *make_submission.py*.
```
test_val_filenames = ['inferences/inceptionv3.0.test_val.csv',
                      'inferences/se_resnext50.0.test_val.csv']
                      
test_filenames = ['inferences/inceptionv3.0.test.csv',
                  'inferences/se_resnext50.0.test.csv']
                  
weights = [1.0, 1.0]
```
The command generate two files. One for original submission and the other is modified using data leak.
- submissions/submission.csv
- submissions/submission.csv.leak.csv

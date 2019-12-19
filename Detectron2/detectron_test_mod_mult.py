import torch, torchvision
torch.__version__

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
from detectron2.utils.visualizer import ColorMode

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
coco = COCO("pascal_train.json") # load training annotations
coco2 = COCO("test.json") # load training annotations

from utils import binary_mask_to_rle

# use get dicts file to generate dictionary for catalog
def get_voc_test_dicts(img_dir):
    img_ids = list(coco2.imgs.keys())
    dataset_dicts = []
    for img_id in img_ids: # run through all images
        record = {} # one record per image
        objs = []
        record["file_name"] = os.path.join("test_images/", coco2.imgs[img_id]['file_name'])
        record["width"] = coco2.imgs[img_id]['width']
        record["height"] = coco2.imgs[img_id]['height']
        record["image_id"] = img_id
        annids = coco2.getAnnIds(imgIds=img_id)
        anns = coco2.loadAnns(annids)
        for i in range(len(annids)): # combine into single annotation format
            obj = {
                "bbox": [0,0,0,0],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [0,0,0,0],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# register catlogs for the model
from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["test"]: # total 20 classes, name for visulization purpose(removed)
    DatasetCatalog.register("voc_" + d, lambda d=d: get_voc_test_dicts(d))
    MetadataCatalog.get("voc_" + d).set(thing_classes=["aeroplane",
    "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"])
voc_test_metadata = MetadataCatalog.get("voc_train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

''' set up config file(below) same as training '''
cfg = get_cfg() # may inherit from file you set
cfg.merge_from_file("./configs/MISC/cascade_mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ()
cfg.DATALOADER.NUM_WORKERS = 0 # not to OOM / break pipe
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 80000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20 # total 20 classes

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # where the weight locate

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("voc_test", )
dataset_dicts = get_voc_test_dicts("test") # load the test data as set in get_dict

train_iter = 50000 
''' set up config file(above) same as training '''

# Run through multiple loops to generate submission files
# for all check points.
for i in range(int(train_iter / 5000)):
    cur_iter = ((i + 1) * 5000) - 1
    if(cur_iter < 10000): # way the checkpoint save
        cur_name = "model_000" + str(cur_iter) +".pth"
    else:
        cur_name = "model_00" + str(cur_iter) + ".pth"
    print(cur_name)
    cfg.MODEL.WEIGHTS = os.path.join("output/", cur_name) # already 10000
    predictor = DefaultPredictor(cfg)
    coco_dt = []
    for d in dataset_dicts: # each dict in single image      
        im = cv2.imread(d["file_name"])
        outputs = predictor(im) # get predictiopn
        instance_len = len(outputs["instances"].to("cpu").scores.numpy())
        score_nd = outputs["instances"].to("cpu").scores.numpy()
        class_nd = outputs["instances"].to("cpu").pred_classes.numpy()
        mask_nd = np.asarray(outputs["instances"].to("cpu").pred_masks)
        for j in range(instance_len): # each segementation for one dictiuonary
            pred = {}
            pred['image_id'] = int(d["image_id"])
            pred['category_id'] = int(class_nd[j]) + 1 # add back from 0-19 to 1-20
            pred['segmentation'] = binary_mask_to_rle(mask_nd[j])
            pred['score'] = float(score_nd[j])
            coco_dt.append(pred)
    with open("submission_" + str(i) + ".json", "w") as f: # output multiple predictions 
        json.dump(coco_dt, f)
    
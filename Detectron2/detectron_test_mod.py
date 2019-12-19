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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
coco = COCO("pascal_train.json") # load training annotations
coco2 = COCO("test.json") # load training annotations

from utils import binary_mask_to_rle

# write a function that loads the dataset into detectron2's standard format
def get_voc_dicts(img_dir):
    img_ids = list(coco.imgs.keys())
    dataset_dicts = []
    for img_id in img_ids:
        record = {}
        objs = []
        record["file_name"] = os.path.join(img_dir + "/", coco.imgs[img_id]['file_name'])
        record["width"] = coco.imgs[img_id]['width']
        record["height"] = coco.imgs[img_id]['height']
        record["image_id"] = img_id
        annids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annids)
        for i in range(len(annids)):
            obj = {
                "bbox": anns[i]['bbox'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": anns[i]['segmentation'],
                "category_id": anns[i]['category_id'],
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_voc_test_dicts(img_dir):
    img_ids = list(coco2.imgs.keys())
    dataset_dicts = []
    for img_id in img_ids:
        record = {}
        objs = []
        record["file_name"] = os.path.join("test_images/", coco2.imgs[img_id]['file_name'])
        record["width"] = coco2.imgs[img_id]['width']
        record["height"] = coco2.imgs[img_id]['height']
        record["image_id"] = img_id
        annids = coco2.getAnnIds(imgIds=img_id)
        anns = coco2.loadAnns(annids)
        for i in range(len(annids)):
            obj = {
                "bbox": [0,0,0,0],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [0,0,0,0],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("voc_" + d, lambda d=d: get_voc_dicts(d))
    MetadataCatalog.get("voc_" + d).set(thing_classes=["background", "aeroplane",
    "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"])
voc_metadata = MetadataCatalog.get("voc_train")
for d in ["test"]:
    DatasetCatalog.register("voc_" + d, lambda d=d: get_voc_test_dicts(d))
    MetadataCatalog.get("voc_" + d).set(thing_classes=["background", "aeroplane",
    "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"])
voc_test_metadata = MetadataCatalog.get("voc_train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("voc_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0 # not to OOM / break pipe
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # already 10000
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 80000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20 + 1  # only has 20 class (ballon)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("voc_test", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_voc_test_dicts("test")

coco_dt = []
for d in dataset_dicts:    
    print(d["file_name"])    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    instance_len = len(outputs["instances"].to("cpu").scores.numpy())
    score_nd = outputs["instances"].to("cpu").scores.numpy()
    class_nd = outputs["instances"].to("cpu").pred_classes.numpy()
    mask_nd = np.asarray(outputs["instances"].to("cpu").pred_masks)
    for i in range(instance_len):
        pred = {}
        pred['image_id'] = int(d["image_id"])
        print("class")
        print(class_nd[i])
        pred['category_id'] = int(class_nd[i])
        print("mask")
        print(binary_mask_to_rle(mask_nd[i]))
        pred['segmentation'] = binary_mask_to_rle(mask_nd[i])
        print("score")
        print(score_nd[i])
        pred['score'] = float(score_nd[i])
        coco_dt.append(pred)
    v = Visualizer(im[:, :, ::-1],
                   metadata=voc_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("result", (v.get_image()[:, :, ::-1]))
    cv2.waitKey(0)

with open("submission.json", "w") as f:
    json.dump(coco_dt, f)
import torch
import torchvision
import detectron2
import os
import json
import itertools
import cv2
import random
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

torch.__version__
setup_logger()
coco = COCO("pascal_train.json")  # load training annotations


def get_voc_dicts(img_dir):  # load train, vallidation data
    img_ids = list(coco.imgs.keys())  # get all images id in json file
    dataset_dicts = []
    for img_id in img_ids:  # run through all images
        record = {}
        objs = []
        record["file_name"] = os.path.join(
            img_dir + "/", coco.imgs[img_id]['file_name'])  # image direcory
        record["width"] = coco.imgs[img_id]['width']
        record["height"] = coco.imgs[img_id]['height']
        record["image_id"] = img_id
        annids = coco.getAnnIds(imgIds=img_id)  # get mask and bbox information
        anns = coco.loadAnns(annids)
        for i in range(len(annids)):
            obj = {
                "bbox": anns[i]['bbox'],
                # correct format of bounding box for this task
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": anns[i]['segmentation'],
                # for remapping to 0-19
                "category_id": anns[i]['category_id'] - 1,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)  # append aone record fopr one image
    return dataset_dicts


# load dict into catlog
# also load class name into catlog
for d in ["train", "val"]:
    DatasetCatalog.register("voc_" + d, lambda d=d: get_voc_dicts(d))
    MetadataCatalog.get("voc_" + d).set(thing_classes=[
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
voc_metadata = MetadataCatalog.get("voc_train")

dataset_dicts = get_voc_dicts("train")
for d in random.sample(dataset_dicts, 5):  # show 5 image for debugging
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=voc_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("res", (vis.get_image()[:, :, ::-1]))
    cv2.waitKey(0)

''' set up config file(below) same as testing '''
cfg = get_cfg()
# use cascade mrcnn config file
cfg.merge_from_file("./configs/MISC/cascade_mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("voc_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0  # not to OOM / break pipe
# can also uncomment this line to use default setting
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
# uncomment to train on your own pretrain
# cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50000  # how much iteration you want to run
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
''' set up config file(above) same as testing '''

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)  # reload
trainer.train()

# Look at training curves in tensorboard:
# tensorboard --logdir output

''' model may be save in output directory '''
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("voc_val", )
predictor = DefaultPredictor(cfg)

''' visualize validation datas '''
# tewt at validation set (currently same as train)
dataset_dicts = get_voc_dicts("val")
for d in random.sample(dataset_dicts, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=voc_metadata,
                   scale=0.8,
                   # remove colors for unsegments
                   instance_mode=ColorMode.IMAGE_BW
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("result", (v.get_image()[:, :, ::-1]))
    cv2.waitKey(0)

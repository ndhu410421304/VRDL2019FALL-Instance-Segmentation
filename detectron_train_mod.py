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

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("voc_" + d, lambda d=d: get_voc_dicts(d))
    MetadataCatalog.get("voc_" + d).set(thing_classes=["background", "aeroplane",
    "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"])
voc_metadata = MetadataCatalog.get("voc_train")

"""To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:"""

dataset_dicts = get_voc_dicts("train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=voc_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("res",(vis.get_image()[:, :, ::-1]))
    cv2.waitKey(0)

"""## Train!

Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the voc dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU, or ~2 minutes on a P100 GPU.
"""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("voc_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0 # not to OOM / break pipe
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"  # initialize from model zoo
# cfg.MODEL.WEIGHTS = "pretrain/R-50-GN.pkl"  # initialize from model zoo
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # already 10000
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 80000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20 + 1  # only has 20 class (ballon)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False) # reload
trainer.train()

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""## Inference & evaluation using the trained model
Now, let's run inference with the trained model on the voc validation dataset. First, let's create a predictor using the model we just trained:
"""

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("voc_val", )
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_voc_dicts("val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=voc_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("result", (v.get_image()[:, :, ::-1]))
    cv2.waitKey(0)

'''
"""We can also evaluate its performance using AP metric implemented in COCO API.
This gives an AP of ~70%. Not bad!
"""

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("voc_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "voc_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
'''
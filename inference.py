# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger('inference_log')

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

im=cv2.imread('datasets/test/21_0.jpg')

'''
Read Metadata from file
'''
with open('metadata.json') as f:
	dataset_metadata = json.load(f)

'''
Or you can create a Metadata , both will work the same.
'''
#from detectron2.data.catalog import Metadata
#dataset_metadata = Metadata()
#dataset_metadata.set(thing_classes = ['head', 'shrimp', 'tail'])

'''
Set config, make sure they are the same as what you used during training.
'''
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file("ori_config.yml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = ("model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

predictor = DefaultPredictor(cfg)
outputs = predictor(im)
print(outputs)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], metadata = dataset_metadata, scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('res',out.get_image()[:, :, ::-1])
cv2.waitKey(0)

'''
python3 demo.py --config-file /home/lemon/Desktop/config.yml --input /home/lemon/Desktop/datasets/test/1_0.jpg /home/lemon/Desktop/datasets/test/2_0.jpg /home/lemon/Desktop/datasets/test/3_0.jpg --opts MODEL.DEVICE cpu MODEL.WEIGHTS /home/lemon/Desktop/output/model_final.pth
'''
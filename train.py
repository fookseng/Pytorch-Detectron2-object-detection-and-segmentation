# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# Write log to a folder called 'log'
setup_logger('log')

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy
import torch
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader

'''
Function to apply data augmentations
'''
def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.Resize((800,600)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

'''
Define our custom Trainer
'''
class CocoTrainer(DefaultTrainer):
    @classmethod
    # For evaluation purpose
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    @classmethod
    # For data augmentations
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

'''
Register our dataset
'''
#DatasetCatalog.remove("py_dataset_val")
register_coco_instances("py_dataset_train", {}, "datasets/train.json", "datasets/train/")
register_coco_instances("py_dataset_valid", {}, "datasets/valid.json", "datasets/valid/")

'''
Visualize annotations to make sure we have registered the dataset correctly.
'''
dataset_dicts = DatasetCatalog.get("py_dataset_train")
py_dataset_train_metadata = MetadataCatalog.get("py_dataset_train")

'''
Save a copy of metadata for future inference usage.
'''
with open('metadata.json', 'w') as outfile:
    json.dump(py_dataset_train_metadata.as_dict(), outfile)

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=py_dataset_train_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Visualize annotation", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

'''
Modify the configurations for training.
You may modify the following configurations.
'''
cfg = get_cfg()

# Comment this line if you do have a GPU available.
cfg.MODEL.DEVICE='cpu'

# Select our model
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# The name of your registered dataset
cfg.DATASETS.TRAIN = ("py_dataset_train",)
cfg.DATASETS.TEST = ("py_dataset_valid",)

cfg.DATALOADER.NUM_WORKERS = 2

# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2

# Pick a good Learning Rate
cfg.SOLVER.BASE_LR = 0.00025  

#cfg.SOLVER.WARMUP_ITERS = 1000

# 2000 iterations seems good enough for this dataset; you will need to train longer for a practical dataset
# Increase the value if val mAP is rising or decrease if overfit.
cfg.SOLVER.MAX_ITER = 3000

# Do not decay learning rate
cfg.SOLVER.STEPS = []  
# The learning rate will be reduced by GAMMA
#cfg.SOLVER.STEPS = (1000, 1500)
#cfg.SOLVER.GAMMA = 0.05

# (default: 512) Lower your batch size if you encounter the error "run out of memory".
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   

# The number of classes in your datasets. Here, we have only has 3 classes(head, tail, shrimp). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: This config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

# No. of iterations after which the Validation Set is evaluated.
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

'''
Save config file for inference usage.
'''
f = open('config.yml', 'w')
f.write(cfg.dump())
f.close()

'''
Start our training by using the custom Trainer
'''
#trainer = DefaultTrainer(cfg) 
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

'''
Evaluate using Valid Set after we have done the training.
'''
evaluator = COCOEvaluator("py_dataset_valid", ("bbox", "segm"), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "py_dataset_valid")
print(inference_on_dataset(trainer.model, val_loader, evaluator))




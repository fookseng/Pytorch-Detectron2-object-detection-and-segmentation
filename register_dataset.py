from detectron2.structures import BoxMode
from cv2 import cv2
import os
import json

def get_shrimp_dicts(img_dir, json_dir, category_id):
    dataset_dicts = []
    for idx, file in enumerate(os.listdir(json_dir)):
        if file.endswith('.json'):
            with open(os.path.join(json_dir, file)) as f:
                v = json.load(f)
                record = {}
                filename = os.path.join(img_dir, v["imagePath"])
                height, width = cv2.imread(filename).shape[:2]
                
                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width
                
                annos = v["shapes"]
                objs = []
                
                for anno in annos:
                    points = anno["points"]
                    px = []
                    py = []
                    for point in points:
                        px.append(point[0])
                        py.append(point[1])
                    poly = [ (x + 0.5, y + 0.5) for x, y in zip(px, py) ]
                    poly = [ p for x in poly for p in x]
                    
                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": category_id[anno["label"]],
                    }
                    
                    objs.append(obj)
                
                record["annotations"] = objs
                dataset_dicts.append(record)
    
    return dataset_dicts

from detectron2.data import MetadataCatalog, DatasetCatalog

category_id = {
    "shrimp": 0,
    "head": 1,
    "tail": 2
}

TRAIN_ROOT = 'dataset/shrimp/train/'
TEST_ROOT = 'dataset/shrimp/test/'

DatasetCatalog.register("shrimp-train", lambda: get_shrimp_dicts(TRAIN_ROOT+'img', TRAIN_ROOT+'json', category_id))
DatasetCatalog.register("shrimp-test", lambda: get_shrimp_dicts(TEST_ROOT+'img', TEST_ROOT+'json', category_id))
MetadataCatalog.get("shrimp-train").set(thing_classes=["shrimp", "head","tail"])
MetadataCatalog.get("shrimp-test").set(thing_classes=["shrimp", "head","tail"])
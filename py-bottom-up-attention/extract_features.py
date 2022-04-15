# Adapted from code in https://github.com/airsplay/py-bottom-up-attention
# Please see https://github.com/airsplay/py-bottom-up-attention/blob/master/LICENSE for the Apache 2.0 License of that code

import os
import sys
import io
import json
from tqdm import tqdm

import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import torch

NUM_OBJECTS = 36

from torch import nn

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances


# Load VG Classes
data_path = 'data/genome/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())


MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs

cfg = get_cfg()
cfg.merge_from_file("configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
# cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
predictor = DefaultPredictor(cfg)

def doit(raw_image, raw_boxes):
    # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        print(pred_class_logits.shape)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes
        )
        return instances, roi_features

# Image root
img_root = sys.argv[1]
# jsonl input file
f = open(sys.argv[2])
lines = f.readlines()
data = [json.loads(line) for line in lines]
if len(sys.argv) > 4:
    # Predicted boxes JSON file
    f = open(sys.argv[3])
    predicted_boxes = json.load(f)
boxes_dict = {}
for datum in tqdm(data):
    if 'coco' in datum['file_name'].lower():
        datum['file_name'] = '_'.join(datum['file_name'].split('_')[:-1])+'.jpg'
    im = cv2.imread(img_root+datum['file_name'])
    if len(sys.argv) == 4:
        boxes = np.array(
            [[ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]] for ann in datum['anns']]
        )
        conf = np.array([1 for _ in datum["anns"]])
    else:
        assert len(sys.argv) > 4
        conf = np.array(predicted_boxes[str(datum["image_id"])]["scores"])
        print(conf.shape)
        if len(conf) > 0:
            boxes = np.array(predicted_boxes[str(datum["image_id"])]["boxes"])[(-conf).argsort(),:]
            conf = conf[(-conf).argsort()]
        else:
            boxes = np.array([[0, 0, im.shape[1], im.shape[0]]])
            conf = np.array([1.])
    _, feats = doit(im, boxes)
    boxes_dict[datum['image_id']] = {"boxes": torch.from_numpy(boxes), "features": feats, "width": im.shape[1], "height": im.shape[0], "conf": torch.from_numpy(conf)}
# Output .pt file
torch.save(boxes_dict, sys.argv[4])

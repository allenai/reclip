import json
import torch
import argparse
from tqdm import tqdm

PERSON_LABEL = 134
CITYSCAPES_LABELS = {13, 125, 134, 136, 146, 151, 157, 169, 181, 219, 230, 231, 233, 246, 285, 334, 337, 406, 452, 471, 480, 596, 607, 695}

parser = argparse.ArgumentParser()
parser.add_argument("--input_path")
parser.add_argument("--output_path")
args = parser.parse_args()

f = open(args.input_path)
lines = f.readlines()
data = [json.loads(line) for line in lines]
detections = {}
for datum in tqdm(data):
    parts = datum['file_name'].split('/')
    file_name = parts[0]+'/'+parts[-1].split('.')[0]+'.pt'
    print(file_name)
    instances = torch.load(file_name, map_location='cpu')
    indices = [i for i in range(len(instances['instances'].pred_classes)) if instances['instances'].pred_classes[i].item() == PERSON_LABEL]
    # indices = list(range(len(instances['instances'].pred_classes)))
    detections[datum['image_id']] = {"boxes": instances['instances'].pred_boxes.tensor[indices,:].tolist(), "scores": instances['instances'].scores[indices].tolist()}
fout = open(args.output_path, 'w')
json.dump(detections, fout)
fout.close()

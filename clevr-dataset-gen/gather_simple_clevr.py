import json
import random
import argparse
from copy import deepcopy
from collections import defaultdict

from bounding_box import extract_bounding_boxes

def construct_non_spatial_text(object_list):
    text = ""
    for obj in object_list[:-1]:
        text += "a "+obj
        if len(object_list) > 2:
            text += ","
        text += " "
    text += "and a "+object_list[-1]
    return text

parser = argparse.ArgumentParser()
parser.add_argument('--scenes_path', help="Path to scenes file")
parser.add_argument("--output_path", help="Path to output file")
parser.add_argument('--spatial', action='store_true', help="If true, collect pairs of scenes with same objects in different spatial configuration")
parser.add_argument('--max_num_objects', type=int, default=3, help="Max number of objects in the filtered scenes")
parser.add_argument('--mode', type=str, default='text_pair', choices=['text_pair', 'image_pair'])
args = parser.parse_args()

with open(args.scenes_path) as scenes_file:
    scenes = json.load(scenes_file)
    if isinstance(scenes, dict):
        scenes = scenes["scenes"]
for scene in scenes:
    xmin, ymin, xmax, ymax, _, _ = extract_bounding_boxes(scene)
    scene["boxes"] = [[xmin[i], ymin[i], xmax[i], ymax[i]] for i in range(len(xmin))]
scenes_by_type = defaultdict(list)
all_colors = set()
all_shapes = set()
for scene in scenes:
    if len(scene["objects"]) <= args.max_num_objects:
        types = sorted([obj["color"]+" "+obj["shape"] for obj in scene["objects"]])
        scenes_by_type[",".join(types)].append(scene)
    for obj in scene["objects"]:
        all_colors.add(obj["color"])
        all_shapes.add(obj["shape"])
all_colors = list(all_colors)
all_shapes = list(all_shapes)
examples = []
if args.spatial:
    for scene_type in scenes_by_type:
        if args.mode == "image_pair":
            # assert len(scenes_by_type[scene_type]) > 1
            scene1 = random.choice(scenes_by_type[scene_type])
            print(len(scenes_by_type[scene_type]))
            scene2 = scene1
            diff_rels = []
            random.shuffle(scenes_by_type[scene_type])
            i = 0
            while len(diff_rels) == 0 and i < len(scenes_by_type[scene_type]):
                scene2 = scenes_by_type[scene_type][i]
                obj_index_map = {0: 0, 1: 1}
                if scene2["objects"][0]["color"] != scene1["objects"][0]["color"] or scene2["objects"][0]["shape"] != scene1["objects"][0]["shape"]:
                    scene2["objects"] = scene2["objects"][::-1]
                    for rel in scene2["relationships"]:
                        scene2["relationships"][rel] = scene2["relationships"][rel][::-1]
                        for j in range(len(scene2['relationships'][rel])):
                            if len(scene2['relationships'][rel][j]) > 0:
                                scene2['relationships'][rel][j][0] = 1-scene2['relationships'][rel][j][0]
                assert scene2["objects"][obj_index_map[0]]["color"] == scene1["objects"][0]["color"] and scene2["objects"][obj_index_map[0]]["shape"] == scene2["objects"][obj_index_map[0]]["shape"]
                for rel in scene1["relationships"]:
                    if scene1["relationships"][rel] != scene2["relationships"][rel]:
                        diff_rels.append(rel)
                i += 1
            if len(diff_rels) == 0:
                continue
            print(diff_rels)
            example = deepcopy(scene1)
            example["image_filename2"] = scene2["image_filename"]
            rel = random.choice(diff_rels)
            if rel in {"left", "right"}:
                if len(scene1["relationships"][rel][0]) != 1:
                    obj_index_map = {1: 0, 0: 1}
                example["text1"] = "a "+scene1["objects"][obj_index_map[1]]["color"]+" "+scene1["objects"][obj_index_map[1]]["shape"]+" to the "+rel+" of a "+scene1["objects"][obj_index_map[0]]["color"]+" "+scene1["objects"][obj_index_map[0]]["shape"]+"."
            else:
                if len(scene1["relationships"][rel][0]) != 1:
                    obj_index_map = {1: 0, 0: 1}
                if rel == "front":
                    example["text1"] = "a "+scene1["objects"][obj_index_map[1]]["color"]+" "+scene1["objects"][obj_index_map[1]]["shape"]+" in front of a "+scene1["objects"][obj_index_map[0]]["color"]+" "+scene1["objects"][obj_index_map[0]]["shape"]+"."
                else:
                    example["text1"] = "a "+scene1["objects"][obj_index_map[1]]["color"]+" "+scene1["objects"][obj_index_map[1]]["shape"]+" behind a "+scene1["objects"][obj_index_map[0]]["color"]+" "+scene1["objects"][obj_index_map[0]]["shape"]+"."
            examples.append(example)
        if args.mode == "text_pair":
            scene = random.choice(scenes_by_type[scene_type])
            object_order = list(range(len(scene["objects"])))
            random.shuffle(object_order)
            relation_order = list(scene["relationships"].keys())
            random.shuffle(relation_order)
            added_example = False
            for obj in object_order:
                for relation in relation_order:
                    if len(scene["relationships"][relation][obj]) > 0:
                        obj2 = random.choice(scene["relationships"][relation][obj])
                        example = deepcopy(scene)
                        left_right = ["left", "right"]
                        if relation in left_right:
                            example["text1"] = "a "+scene["objects"][obj2]["color"]+" "+scene["objects"][obj2]["shape"]+" to the "+relation+" of a "+scene["objects"][obj]["color"]+" "+scene["objects"][obj]["shape"]+"."
                            example["text2"] = "a "+scene["objects"][obj2]["color"]+" "+scene["objects"][obj2]["shape"]+" to the "+left_right[1-left_right.index(relation)]+" of a "+scene["objects"][obj]["color"]+" "+scene["objects"][obj]["shape"]+"."
                        elif relation == "front":
                            example["text1"] = "a "+scene["objects"][obj2]["color"]+" "+scene["objects"][obj2]["shape"]+" in front of a "+scene["objects"][obj]["color"]+" "+scene["objects"][obj]["shape"]+"."
                            example["text2"] = "a "+scene["objects"][obj2]["color"]+" "+scene["objects"][obj2]["shape"]+" behind a "+scene["objects"][obj]["color"]+" "+scene["objects"][obj]["shape"]+"."
                        elif relation == "behind":
                            example["text1"] = "a "+scene["objects"][obj2]["color"]+" "+scene["objects"][obj2]["shape"]+" behind a "+scene["objects"][obj]["color"]+" "+scene["objects"][obj]["shape"]+"."
                            example["text2"] = "a "+scene["objects"][obj2]["color"]+" "+scene["objects"][obj2]["shape"]+" in front of a "+scene["objects"][obj]["color"]+" "+scene["objects"][obj]["shape"]+"."
                        examples.append(example)
                        added_example = True
                        break
                if added_example:
                    break
else:
    if args.mode == "image_pair":
        for scene_type in scenes_by_type:
            scene = deepcopy(random.choice(scenes_by_type[scene_type]))
            scene2_type = scene_type
            while scene2_type == scene_type:
                scene2_type = random.choice(list(scenes_by_type.keys()))
            scene2 = random.choice(scenes_by_type[scene2_type])
            scene['image_filename2'] = scene2['image_filename']
            scene['text1'] = construct_non_spatial_text(scene_type.split(","))
            examples.append(scene)
    if args.mode == "text_pair":
        for scene_type in scenes_by_type:
            scene = random.choice(scenes_by_type[scene_type])
            objects_in_scene = scene_type.split(",")
            random.shuffle(all_colors)
            random.shuffle(all_shapes)
            missing_object = None
            for color in all_colors:
                for shape in all_shapes:
                    if color+" "+shape not in objects_in_scene:
                        missing_object = color+" "+shape
                        break
                if missing_object is not None:
                    break
            example = deepcopy(scene)
            random.shuffle(objects_in_scene)
            example["text1"] = construct_non_spatial_text(objects_in_scene)
            objects_in_scene[random.choice(list(range(len(objects_in_scene))))] = missing_object
            example["text2"] = construct_non_spatial_text(objects_in_scene)
            examples.append(example)
fout = open(args.output_path, 'w')
json.dump(examples, fout)
fout.close()
print(len(examples))

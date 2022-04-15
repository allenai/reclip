"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Referring Expression dataset
"""
import random
import numpy as np
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from .data import (DetectFeatLmdb,
                   pad_tensors, get_gather_index)
from pytorch_pretrained_bert import BertTokenizer

class ReTxtTokJson(object):
    def __init__(self, db_dir, max_txt_len=120):
        f = open(db_dir)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.refs_dict = {}
        for index, datum in enumerate(data):
            for sent_index, sent in enumerate(datum['sentences']):
                self.refs_dict[(index, sent_index)] = (datum, sent)
        self.sent_ids = list(self.refs_dict.keys())

    def __len__(self):
        return len(self.sent_ids)

    def shuffle(self):
        # we shuffle ref_ids and make sent_ids according to ref_ids
        random.shuffle(self.sent_ids)

    def __getitem__(self, id_):
        # sent_id = self.sent_ids[i]
        txt_dump = self.db[id_]
        return txt_dump

class ReTrainJsonDataset(object):
    def __init__(self, txt_db, img_db):
        self.txt_db = txt_db
        self.img_db = img_db
        self.tokenizer = self.txt_db.tokenizer
    def __len__(self):
        return len(self.txt_db)
    def __getitem__(self, i):
        ref_id, sent_id = self.txt_db.sent_ids[i]
        datum = self.txt_db.data[ref_id]
        sentence = datum['sentences'][sent_id]['raw']
        tokens = self.tokenizer.tokenize(sentence)
        token_ids = torch.tensor([self.tokenizer.vocab['[CLS]']]+self.tokenizer.convert_tokens_to_ids(tokens)+[self.tokenizer.vocab['[SEP]']], dtype=torch.long)
        feats, boxes, width, height = self.img_db[datum['image_id']]
        attn_masks = torch.ones(len(token_ids)+len(boxes), dtype=torch.long)
        obj_masks = torch.tensor([0]*len(boxes), dtype=torch.bool) # torch.uint8)
        width_height = torch.cat((boxes[:,2:3]-boxes[:,0:1], boxes[:,3:4]-boxes[:,1:2]), dim=1)
        pos_feat = torch.cat((boxes, width_height, width_height[:,:1]*width_height[:,1:]), dim=1).float()
        pos_feat[:,[0,2,4,6]] = pos_feat[:,[0,2,4,6]] / float(width)
        pos_feat[:,[1,3,5,6]] = pos_feat[:,[1,3,5,6]] / float(height)
        assert pos_feat.shape[1] == 7
        boxes = torch.cat((boxes[:,:2], width_height), dim=1)
        # print(boxes)
        gold_index = [j for j in range(len(datum['anns'])) if datum['anns'][j]['id'] == datum['ann_id']][0]
        # print(datum['anns'][gold_index])
        gold_box = torch.tensor(datum["anns"][gold_index]["bbox"], dtype=torch.float32)
        feats = feats.float()
        pos_feat = pos_feat.float()
        boxes = boxes.float()
        return (token_ids, feats, pos_feat, attn_masks, obj_masks, torch.tensor([gold_index]))

    # IoU function
    def computeIoU(self, box1, box2):
        # each box is of [x1, y1, w, h]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
        inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return float(inter)/union

    def shuffle(self):
        self.txt_db.shuffle()

def re_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L) padded with 0
    :position_ids  : (n, max_L) padded with 0
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, feat_dim)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max_{L+num_bb}) padded with 0
    :obj_masks     : (n, max_num_bb) padded with 1
    :targets       : (n, )
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, obj_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    obj_masks = pad_sequence(
        obj_masks, batch_first=True, padding_value=1)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    return {'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'obj_masks': obj_masks,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'targets': targets,
            'txt_lens': txt_lens,
            'num_bbs': num_bbs}

class ReEvalJsonDataset(object):
    def __init__(self, txt_db, img_db):
        self.txt_db = txt_db
        self.img_db = img_db
        self.tokenizer = self.txt_db.tokenizer
    def __len__(self):
        return len(self.txt_db)
    def __getitem__(self, i):
        ref_id, sent_id = self.txt_db.sent_ids[i]
        datum = self.txt_db.data[ref_id]
        sentence = datum['sentences'][sent_id]['raw']
        tokens = self.tokenizer.tokenize(sentence)
        token_ids = torch.tensor([self.tokenizer.vocab['[CLS]']]+self.tokenizer.convert_tokens_to_ids(tokens)+[self.tokenizer.vocab['[SEP]']], dtype=torch.long)
        feats, boxes, width, height = self.img_db[datum['image_id']]
        attn_masks = torch.ones(len(token_ids)+len(boxes), dtype=torch.long)
        obj_masks = torch.tensor([0]*len(boxes), dtype=torch.uint8)
        width_height = torch.cat((boxes[:,2:3]-boxes[:,0:1], boxes[:,3:4]-boxes[:,1:2]), dim=1)
        pos_feat = torch.cat((boxes, width_height, width_height[:,:1]*width_height[:,1:]), dim=1).float()
        pos_feat[:,[0,2,4,6]] = pos_feat[:,[0,2,4,6]] / float(width)
        pos_feat[:,[1,3,5,6]] = pos_feat[:,[1,3,5,6]] / float(height)
        assert pos_feat.shape[1] == 7
        boxes = torch.cat((boxes[:,:2], width_height), dim=1)
        # print(boxes)
        gold_index = [j for j in range(len(datum['anns'])) if datum['anns'][j]['id'] == datum['ann_id']][0]
        # print(datum['anns'][gold_index])
        gold_box = torch.tensor(datum["anns"][gold_index]["bbox"], dtype=torch.float32)
        feats = feats.float()
        pos_feat = pos_feat.float()
        boxes = boxes.float()
        return (token_ids, feats, pos_feat, attn_masks, obj_masks, gold_box, boxes, i)

    # IoU function
    def computeIoU(self, box1, box2):
        # each box is of [x1, y1, w, h]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
        inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return float(inter)/union


def re_eval_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L)
    :position_ids  : (n, max_L)
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, d)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max{L+num_bb})
    :obj_masks     : (n, max_num_bb)
    :tgt_box       : list of n [xywh]
    :obj_boxes     : list of n [[xywh, xywh, ...]]
    :sent_ids      : list of n [sent_id]
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, obj_masks,
     tgt_box, obj_boxes, sent_ids) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    obj_masks = pad_sequence(
        obj_masks, batch_first=True, padding_value=1)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    return {'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'obj_masks': obj_masks,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'tgt_box': tgt_box,
            'obj_boxes': obj_boxes,
            'sent_ids': sent_ids,
            'txt_lens': txt_lens,
            'num_bbs': num_bbs}

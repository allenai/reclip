"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""
from collections import defaultdict
from contextlib import contextmanager
import io
import json
from os.path import exists

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import horovod.torch as hvd
from tqdm import tqdm
import lmdb

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32)
           if arr.dtype == np.float16 else arr
           for k, arr in feat_dict.items()}
    return out


def compute_num_bb(confs, conf_th, min_bb, max_bb):
    num_bb = max(min_bb, (confs > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return num_bb


def _check_distributed():
    try:
        dist = hvd.size() != hvd.local_size()
    except ValueError:
        # not using horovod
        dist = False
    return dist

class DetectFeatPt(object):
    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36,
                 compress=True):
        self.img_dir = img_dir
        print("Loading boxes and features from "+self.img_dir)
        self.data = torch.load(self.img_dir, map_location='cpu')
        self.conf_th = conf_th
        self.max_bb = max_bb
    def __getitem__(self, image_id):
        data = self.data[image_id]
        boxes = data["boxes"]
        feats = data["features"]
        boxes = boxes[:self.max_bb,:]
        feats = feats[:self.max_bb,:]
        conf = data["conf"][:self.max_bb]
        boxes = boxes[conf >= self.conf_th,:]
        feats = feats[conf >= self.conf_th,:]
        return feats, boxes, data["width"], data["height"]

class DetectFeatLmdb(object):
    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36,
                 compress=True):
        self.img_dir = img_dir
        if conf_th == -1:
            db_name = f'feat_numbb{num_bb}'
            self.name2nbb = defaultdict(lambda: num_bb)
        else:
            db_name = f'feat_th{conf_th}_max{max_bb}_min{min_bb}'
            nbb = f'nbb_th{conf_th}_max{max_bb}_min{min_bb}.json'
            if not exists(f'{img_dir}/{nbb}'):
                # nbb is not pre-computed
                self.name2nbb = None
            else:
                self.name2nbb = json.load(open(f'{img_dir}/{nbb}'))
        self.compress = compress
        if compress:
            db_name += '_compressed'

        if self.name2nbb is None:
            if compress:
                db_name = 'all_compressed'
            else:
                db_name = 'all'
        # only read ahead on single node training
        self.env = lmdb.open(f'{img_dir}/{db_name}',
                             readonly=True, create=False,
                             readahead=not _check_distributed())
        self.txn = self.env.begin(buffers=True)
        if self.name2nbb is None:
            self.name2nbb = self._compute_nbb()

    def _compute_nbb(self):
        name2nbb = {}
        fnames = json.loads(self.txn.get(key=b'__keys__').decode('utf-8'))
        for fname in tqdm(fnames, desc='reading images'):
            dump = self.txn.get(fname.encode('utf-8'))
            if self.compress:
                with io.BytesIO(dump) as reader:
                    img_dump = np.load(reader, allow_pickle=True)
                    confs = img_dump['conf']
            else:
                img_dump = msgpack.loads(dump, raw=False)
                confs = img_dump['conf']
            name2nbb[fname] = compute_num_bb(confs, self.conf_th,
                                             self.min_bb, self.max_bb)

        return name2nbb

    def __del__(self):
        self.env.close()

    def get_dump(self, file_name):
        # hack for MRC
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = _fp16_to_fp32(img_dump)
        else:
            img_dump = msgpack.loads(dump, raw=False)
            img_dump = _fp16_to_fp32(img_dump)
        img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
        return img_dump

    def __getitem__(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = {'features': img_dump['features'],
                            'norm_bb': img_dump['norm_bb']}
        else:
            img_dump = msgpack.loads(dump, raw=False)
        img_feat = torch.tensor(img_dump['features'][:nbb, :]).float()
        img_bb = torch.tensor(img_dump['norm_bb'][:nbb, :]).float()
        return img_feat, img_bb

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index

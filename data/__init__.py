"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import importlib

import numpy as np
import skimage.color as color
import skimage.io as io
import torch

import utils


def generate_loader(phase, opt):
    cname = opt.dataset.replace("_", "")
    if "DIV2K" in opt.dataset:
        mname = importlib.import_module("data.div2k")
    elif "RealSR" in opt.dataset:
        mname = importlib.import_module("data.realsr")
    elif "SR" in opt.dataset: # SR benchmark datasets
        mname = importlib.import_module("data.benchmark")
        cname = "BenchmarkSR"
    elif "DN" in opt.dataset: # DN benchmark datasets
        mname = importlib.import_module("data.benchmark")
        cname = "BenchmarkSR"
    elif "JPEG" in opt.dataset: # JPEG benchmark datasets
        mname = importlib.import_module("data.benchmark")
        cname = "BenchmarkSR"
    elif "UCLand" in opt.dataset:
        mname = importlib.import_module("data.ucland")
        cname = "UCLand"
    else:
        raise ValueError("Unsupported dataset: {}".format(opt.dataset))

    kwargs = {
        "batch_size": opt.batch_size if phase == "train" else opt.test_batch,
        "num_workers": opt.num_workers if phase == "train" else 0,
        "shuffle": phase == "train",
        "drop_last": phase == "train",
    }

    dataset = getattr(mname, cname)(phase, opt)
    return torch.utils.data.DataLoader(dataset, **kwargs)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt):
        print("Load dataset... (phase: {}, len: {})".format(phase, len(self.HQ_paths)))
        # self.HQ, self.LQ = list(), list()
        # for HQ_path, LQ_path in zip(self.HQ_paths, self.LQ_paths):
        #     self.HQ += [io.imread(HQ_path)]
        #     self.LQ += [io.imread(LQ_path)]

        self.phase = phase
        self.opt = opt

    def im2tensor(self, im):
        np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_t).float()
        return tensor

    def get_image(self, idx):
        HQ = io.imread(self.HQ_paths[idx])
        LQ = io.imread(self.LQ_paths[idx])
        if len(HQ.shape) < 3:
            HQ = color.gray2rgb(HQ)
        if len(LQ.shape) < 3:
            LQ = color.gray2rgb(LQ)
        return HQ, LQ

    def __getitem__(self, index):
        # follow the setup of EDSR-pytorch
        if self.phase == "train":
            index = index % len(self.HQ_paths)

        HQ, LQ = self.get_image(index)

        if self.phase == "train":
            inp_scale = HQ.shape[0] // LQ.shape[0]
            HQ, LQ = utils.crop(HQ, LQ, self.opt.patch_size, inp_scale)
            HQ, LQ = utils.flip_and_rotate(HQ, LQ)

        return self.im2tensor(HQ), self.im2tensor(LQ)

    def __len__(self):
        # follow the setup of EDSR-pytorch
        if self.phase == "train":
            return (1000 * self.opt.batch_size) // len(self.HQ_paths) * len(self.HQ_paths)
        return len(self.HQ_paths)

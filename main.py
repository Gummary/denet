"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import os
import json
import utils
import torch
import importlib
from option import get_option
from solver import Solver

def main():
    opt = get_option()
    if opt.save_result:
        save_root = os.path.join(opt.save_root, opt.dataset)
        utils.mkdir_or_exist(save_root)
    utils.mkdir_or_exist(opt.ckpt_root)
    utils.mkdir_or_exist(opt.save_root)
    logger = utils.create_logger(opt)
    torch.manual_seed(opt.seed)

    module = importlib.import_module("model.{}".format(opt.model.lower()))

    if not opt.test_only:
        print(json.dumps(vars(opt), indent=4))

    solver = Solver(module, opt)
    if opt.test_only:
        logger.info("Evaluate {} (loaded from {})".format(opt.model, opt.pretrain))
        psnr = solver.evaluate()
        logger.info("{:.2f}".format(psnr))
    else:
        solver.fit()

if __name__ == "__main__":
    main()

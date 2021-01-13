# Copyright 2021 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.
import importlib
import os

import utils
from option import get_option
from solver import Solver


def main():
    opt = get_option()
    save_root = os.path.join(opt.save_root, opt.dataset)
    utils.mkdir_or_exist(save_root)
    if "_DC" in opt.model:
        module = importlib.import_module("model.dynet")
    else:
        module = importlib.import_module("model.{}".format(opt.model.lower()))
    solver = Solver(module, opt)

    solver.save_mid_result()


if __name__ == '__main__':
    main()

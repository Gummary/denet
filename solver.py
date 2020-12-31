"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import os
import time
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import augments
from data import generate_loader
from torchsummaryX import summary
import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)

class Solver():
    def __init__(self, module, opt):
        self.opt = opt

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt).to(self.dev)
        summary(self.net, torch.zeros(4, 3, 64, 64).to(self.dev))

        if opt.pretrain:
            self.load(opt.pretrain)

        self.loss_fn = nn.L1Loss()
        self.optim = torch.optim.Adam(
            self.net.parameters(), opt.lr,
            betas=(0.9, 0.999), eps=1e-8
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, [1000*int(d) for d in opt.decay.split("-")],
            gamma=opt.gamma,
        )

        if not opt.test_only:
            self.train_loader = generate_loader("train", opt)
        self.test_loader = generate_loader("test", opt)


        self.t1, self.t2 = None, None
        self.best_psnr, self.best_step = 0, 0


    def fit(self):
        opt = self.opt

        self.t1 = time.time()
        for step in range(opt.max_steps):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(self.train_loader)
                inputs = next(iters)

            HR = inputs[0].to(self.dev)
            LR = inputs[1].to(self.dev)

            # match the resolution of (LR, HR) due to CutBlur
            # if HR.size() != LR.size():
            #     scale = HR.size(2) // LR.size(2)
            #     LR = F.interpolate(LR, scale_factor=scale, mode="nearest")

            # HR, LR, mask, aug = augments.apply_augment(
            #     HR, LR,
            #     opt.augs, opt.prob, opt.alpha,
            #     opt.aux_alpha, opt.aux_alpha, opt.mix_p
            # )
            SR = self.net(LR)
            # if aug == "cutout":
            #     SR, HR = SR*mask, HR*mask

            loss = self.loss_fn(SR, HR)
            self.optim.zero_grad()
            loss.backward()

            if opt.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

            self.optim.step()
            self.scheduler.step()

            if (step+1) % opt.log_intervals == 0:
                _step, _max_steps = (step+1)//1000, self.opt.max_steps//1000
                logger.info(f"[{_step}K/{_max_steps}K] {loss.data:.2f}")

            if (step+1) % opt.eval_steps == 0:
                self.summary_and_save(step)


    def summary_and_save(self, step):
        step, max_steps = (step+1)//1000, self.opt.max_steps//1000
        psnr = self.evaluate()
        self.t2 = time.time()

        if psnr >= self.best_psnr:
            self.best_psnr, self.best_step = psnr, step
            self.save(psnr)

        curr_lr = self.scheduler.get_last_lr()
        eta = (self.t2-self.t1) * (max_steps-step) / 3600
        logger.info(f"[{step}K/{max_steps}K] {psnr:.2f} (Best: {self.best_step:.2f} "
                    f"@ {self.best_step}K step) LR: {curr_lr} ETA {eta} Hours")

        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.net.eval()


        psnr = 0
        for i, inputs in enumerate(self.test_loader):
            HR = inputs[0].to(self.dev)
            LR = inputs[1].to(self.dev)

            # match the resolution of (LR, HR) due to CutBlur
            # if HR.size() != LR.size():
            #     scale = HR.size(2) // LR.size(2)
            #     LR = F.interpolate(LR, scale_factor=scale, mode="nearest")

            SR = self.net(LR).detach()
            HR = HR[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
            SR = SR[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()

            if opt.save_result:
                save_root = os.path.join(opt.save_root, opt.dataset)
                save_path = os.path.join(save_root, "{:04d}.png".format(i+1))
                io.imsave(save_path, SR)

            HR = HR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
            SR = SR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
            if opt.eval_y_only:
                HR = utils.rgb2ycbcr(HR)
                SR = utils.rgb2ycbcr(SR)
            psnr += utils.calculate_psnr(HR, SR)

        self.net.train()

        return psnr/len(self.test_loader)

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        if self.opt.strict_load:
            self.net.load_state_dict(state_dict)
            return

        # when to fine-tune the pre-trained model
        own_state = self.net.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data

                try:
                    own_state[name].copy_(param)
                except Exception:
                    # head and tail modules can be different
                    if name.find("head") == -1 and name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}."
                            .format(name, own_state[name].size(), param.size())
                        )
            else:
                raise RuntimeError(
                    "Missing key {} in model's state_dict".format(name)
                )

    def save(self, psnr):
        save_path = os.path.join(self.opt.ckpt_root, f"best_{psnr:.3f}.pt")
        print(f"save best model to {save_path}")
        torch.save(self.net.state_dict(), save_path)

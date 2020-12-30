import os
import os.path as osp
import glob
import data
import utils


class UCLand(data.BaseDataset):

    CLASSES = ("airplane", "buildings", "denseresidential", "freeway", "harbor", "intersection", "overpass", "parkinglot",
               "storagetanks", "tenniscourt", "agricultural", "baseballdiamond", "beach", "chaparral", "forest", "golfcourse",
               "mediumresidential", "mobilehomepark", "river", "runway", "sparseresidential")

    def __init__(self, phase, opt):
        self.root = opt.dataset_root
        self.scale = opt.scale
        if phase == "train":
            self.image_set = osp.join(self.root, "MainSet", "TrainSet")
        else:
            self.image_set = osp.join(self.root, "MainSet", "ValidSet")

        self.gt_folder = osp.join(self.root, "HR_aligned")
        self.lq_folder = osp.join(self.root, "LR", f"X{opt.scale}")
        self.HQ_paths, self.LQ_paths = self._get_paths()
        super().__init__(phase, opt)

    def _get_paths(self):
        id_files = [osp.join(self.image_set, "{}.txt".format(x))
                    for x in UCLand.CLASSES]
        HQ_paths, LQ_paths = [], []
        for id_file in id_files:
            with open(id_file, 'r') as f:
                image_ids = [x.strip() for x in f]
            class_name = osp.basename(id_file).split('.')[0]

            for basename in image_ids:
                HQ_paths.append(osp.join(self.gt_folder, class_name, f'{basename}.tif'))
                LQ_paths.append(osp.join(self.lq_folder,  class_name, f'{basename}.tif'))
        return HQ_paths, LQ_paths

    def __getitem__(self, index):
        # follow the setup of EDSR-pytorch
        if self.phase == "train":
            index = index % len(self.HQ_paths)

        HQ, LQ = self.get_image(index)

        HQ, LQ = utils.flip_and_rotate(HQ, LQ)

        return self.im2tensor(HQ), self.im2tensor(LQ)
from model_zoo.mambair import buildMambaIR
import os
import time
from functools import partial
from typing import Callable
import seaborn
from model_zoo.swinIR import buildSwinIR
from model_zoo.rcan import buildRCAN
from model_zoo.edsr import buildEDSR
from model_zoo.hat import HAT
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
import torch
import torch.nn as nn
from torch import optim as optim
from torchvision import datasets, transforms
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from basicsr.utils.options import dict2str, parse_options
root_path = '../basicsr/options/test/test_MambaIR_SR_x2.yml'
opt, _ = parse_options(root_path, is_train=False)
opt=opt['datasets']['test_4'] # we use the 4-th SR testsets(i.e. Urban100) to visualize ERF.



class PairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl,
                                                  self.task)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;

        if self.task == 'CAR':
            # image range: [0, 255], int., H W 1
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        elif self.task == 'denoising_gray':  # Matlab + OpenCV version
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)
            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        elif self.task == 'denoising_color':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)

        else:
            # image range: [0, 1], float32., H W 3
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



if True:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns

    #   Set figure parameters
    large = 24;
    med = 24;
    small = 24
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white")
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False






# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
    def heatmap(data, camp='RdYlGn', figsize=(10, 10), ax=None, save_path=None):
        plt.figure(figsize=figsize, dpi=40)
        ax = sns.heatmap(data,
                         xticklabels=False,
                         yticklabels=False, cmap=camp,
                         center=0, annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
        plt.savefig(save_path)

    def analyze_erf(args):
        data = args.source
        print(np.max(data))
        print(np.min(data))
        data = args.ALGRITHOM(data + 1)  # the scores differ in magnitude. take the logarithm for better readability
        data = data / np.max(data)  # rescale to [0,1] for the comparability among models
        heatmap(data, save_path=args.heatmap_save)
        print('heatmap saved at ', args.heatmap_save)

    class Args():
        ...

    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM
    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
    analyze_erf(args)


# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def visualize_erf(MODEL: nn.Module = None, num_images=100, data_path="/data2/guohang/dataset/SR/Urban100/LR_bicubic",
                  save_path=f"/tmp/{time.time()}/erf.npy"):
    def get_input_grad(model, samples):
        outputs = model(samples)
        out_size = outputs.size()
        central_point = outputs[:, :, out_size[2] // 2, out_size[3] // 2].sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    def main(args, MODEL: nn.Module = None):
        print("reading from datapath", args.data_path)
        root = args.data_path
        dataset = PairedImageDataset(opt)

        test_loader = data.DataLoader(dataset,batch_size=1,shuffle=False)

        model = MODEL
        model.cuda().eval()

        optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        optimizer.zero_grad()

        for idx,data_sample in enumerate(test_loader):
            if meter.count == args.num_images:
                return meter.avg
            # we set the imhg size to 120X120 due to the GPU memory constrain
            samples = F.interpolate(data_sample['lq'],size=(120,120))
            samples = samples.cuda(non_blocking=True)
            samples.requires_grad = True
            optimizer.zero_grad()
            contribution_scores = get_input_grad(model, samples)
            torch.cuda.empty_cache()
            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print(f'accumulat{idx}')
                meter.update(contribution_scores)

        return meter.avg


    class Args():
        ...

    args = Args()
    args.num_images = num_images
    args.data_path = data_path
    args.save_path = save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return main(args, MODEL)




if __name__ == '__main__':
    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "show/erf")
    kwargs = dict(only_backbone=True, with_norm=False)
    init_model = buildMambaIR()
    ckpt_path = '/data2/guohang/pretrained/classicSRx2.pth' # path to load your pre_trained model weights
    init_model.load_state_dict(torch.load(ckpt_path)['params'])
    save_path = f"./tmp/{time.time()}/erf.npy"
    grad_map = visualize_erf(init_model, save_path=save_path)
    analyze_erf(source=grad_map, dest=f"{showpath}/erf.png")

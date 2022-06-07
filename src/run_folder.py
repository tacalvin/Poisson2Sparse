import argparse
import os
import pickle
import glob
import random

import kornia
from kornia.losses import psnr

import numpy as np
from numpy.core.fromnumeric import resize, transpose
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import transforms
import torchvision.transforms.functional as T
from training import train
from fmd_dataloader import FMDDataset

# from tqdm import tqdm
from tqdm import trange

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
import yaml
import datetime

from PIL import Image

# from utils.denoising_utils import *

torch.manual_seed(123)
np.random.seed(123)




def load_dataset(cfg_experiment):
    # print(cfg_experiment)
    path = cfg_experiment['dataset']['dataset_path']
    # print(path)

    # if we have gt and raw we need to only load gt
    if 'gtandraw' in cfg_experiment['dataset'].keys() and cfg_experiment['dataset']['gtandraw']:
        print("path", path)
        return DataLoader(FMDDataset(path), batch_size=1, shuffle=False)

    transform_list = []
    if cfg_experiment['dataset']['resize']:

        transform_list.append(transforms.Resize((256,256)))
    if cfg_experiment['dataset']['greyscale']:
        transform_list.append(transforms.Grayscale())

    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)
    if 'mnist' in cfg_experiment['dataset'].keys():
        dataset = datasets.MNIST("~", train=False, transform=transform)
        dataset = torch.utils.data.Subset(dataset, np.random.choice(np.arange(len(dataset)), size=32))
        imgs = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        dataset = datasets.ImageFolder(os.path.dirname(path),transform)
        imgs = DataLoader(dataset, batch_size=1, shuffle=False)
        
    return imgs 


#

class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = self.construct_scalar(node)

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)


Loader.add_constructor('!include', Loader.include)

# def p_noise(img, PEAK):

#     return random_noise(img, 'poisson')

# def p_noise(img, PEAK):
#     # print(img, PEAK)
#     Q = np.max(np.max(img)) / PEAK
#     rate = img / Q
#     noisy = np.random.poisson(rate) * Q

#     # print(noisy)
#     # quit()
#     return noisy


def p_noise(img, PEAK):
    img = np.multiply(img, PEAK)
    img = np.random.poisson(img)
    img = np.divide(img, PEAK)
    return img


def g_noise(img, sigma):
    return random_noise(img, 'gaussian', var=float((sigma/255.0)**2))


def gp_noise(img, PEAK, sigma):
    return random_noise(p_noise(img, PEAK), 'gaussian', var=sigma/255)


def corrupt_dataset(imgs, cfg):
    if cfg['noise'] == 'p':
        # print()
        return [np.clip(p_noise(img, cfg['peak']), 0,1.0) for img in imgs]
    elif cfg['noise'] == 'g':
        return [np.clip(g_noise(img, cfg['sigma']), 0, 1.0) for img in imgs]
    elif cfg['noise'] == 'gp':
        return [np.clip(gp_noise(img, cfg['peak'], cfg['sigma']),0,1.0) for img in imgs]
    else:
        return imgs

    # return imgs


def denoise_experiment(cfg):
    # load images
    output_path = cfg['output_dir']
    experiment_cfg = cfg['experiment_cfg']
    imgs = load_dataset(experiment_cfg)

    # if experiment_cfg['pipeline'] == 'nb':
    train(imgs, cfg)
  
    # print(noisy_imgs[0])


def create_result_dir(cfg, dev=True):
    print(cfg)
    path = cfg['output_dir']
    try:
        num_exp = os.listdir(path)
    except:
        num_exp = []
    # print(num_exp)
    curr_dir_id = len(num_exp)
    output_path = os.path.join(path, "{:04d}".format(curr_dir_id+1))
    print(output_path)
    try:
        # os.mkdir(output_path)
        os.makedirs(output_path)
    except:
        pass

# create copy of cfg into dir

    with open(os.path.join(output_path, 'cfg.yaml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    return output_path


def experiment(cfg):
    start_time = datetime.datetime.now()
    print("Begining Experiment {}".format(start_time))
    output_dir = create_result_dir(cfg, dev=cfg['dev'])
    if output_dir is not None:
        cfg['output_dir'] = output_dir

    # run experiment here
    denoise_experiment(cfg)

    end_time = datetime.datetime.now()
    print("Ending Experiment {}".format(end_time))


def main(cfg_path):
    # load path
    # import yaml
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=Loader)
    experiment(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Denoising Experiment')
    parser.add_argument('--cfg_path', help='Path to experiment config')

    args = parser.parse_args()

    main(args.cfg_path)

import argparse
import torch
import torch.nn as nn
import torchvision.io as io
import torchvision.transforms.functional as TF

from tqdm import trange

import os
from neighbor2neighbor import generate_mask_pair, generate_subimages
from model import build_model
from kornia.metrics import psnr
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", default="src/example.yaml", help="Model and Hyperparamter Config"
)
parser.add_argument("--input_path", required=True, help="Path to image to denoise")
parser.add_argument("--output_path", required=True, help="Path to save denoised image")



class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = self.construct_scalar(node)

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)


Loader.add_constructor('!include', Loader.include)


def main(noisy, config, experiment_cfg):
    model = build_model(config)
    device = torch.device("cuda") if torch.cuda.is_avaiilable() else torch.device("cpu")
    model.to(device)
    print(
        "Number of params: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    # optimizer
    if experiment_cfg["optimizer"] == "Adam":
        LR = experiment_cfg["lr"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # psnr_list = []
    # loss_list = []
    # ssims_list = []
    exp_weight = 0.99

    out_avg = None

    noisy_in = noisy

    H = None
    W = None
    if noisy.shape[2] != noisy.shape[3]:
        H = noisy.shape[2]
        W = noisy.shape[3]
        val_size = (max(H, W) + 31) // 32 * 32
        noisy_in = TF.pad(
            noisy,
            (0, 0, val_size - noisy.shape[3], val_size - noisy.shape[2]),
            padding_mode="reflect",
        )

    t = trange(experiment_cfg["num_iter"])
    pll = nn.PoissonNLLLoss(log_input=False, full=True)
    last_net = None
    psrn_noisy_last = 0.0
    for i in t:

        mask1, mask2 = generate_mask_pair(noisy_in)
        with torch.no_grad():
            noisy_denoised = model(noisy_in)
            noisy_denoised = torch.clamp(noisy_denoised, 0.0, 1.0)

        noisy_in_aug = noisy_in.clone()
        # ic(noisy_in_aug.shape, mask1.shape, noisy_in.shape)
        noisy_sub1 = generate_subimages(noisy_in_aug, mask1)
        noisy_sub2 = generate_subimages(noisy_in_aug, mask2)

        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = model(noisy_sub1)
        noisy_output = torch.clamp(noisy_output, 0.0, 1.0)
        noisy_target = noisy_sub2

        Lambda = experiment_cfg["LAM"]
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        if "l1" in experiment_cfg.keys():
            l1_regularization = 0.0
            for param in model.parameters():
                l1_regularization += param.abs().sum()
            total_loss = experiment_cfg["l1"] * l1_regularization
        # else:
        if "poisson_loss" in experiment_cfg.keys():
            loss1 = pll(noisy_output, noisy_target)
            loss2 = F.l1_loss(noisy_output, noisy_target)
            loss1 += loss2
        elif "poisson_loss_only" in experiment_cfg.keys():
            loss1 = pll(noisy_output, noisy_target)
        elif "l1_loss" in experiment_cfg.keys():
            loss1 = F.l1_loss(noisy_output, noisy_target)

        elif "mse" in experiment_cfg.keys():
            loss1 = torch.mean(diff ** 2)
        else:
            loss1 = F.l1_loss(noisy_output, noisy_target)
            # orch.mean(diff**2)
        loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)

        loss = loss1 + loss2
        if "l1" in experiment_cfg.keys():
            loss += total_loss
        loss.backward()

        with torch.no_grad():
            out_full = model(noisy).detach().cpu()
            if H is not None:
                out_full = out_full[:, :, :H, :W]
            if out_avg is None:
                out_avg = out_full.detach().cpu()
            else:
                out_avg = out_avg * exp_weight + out_full * (1 - exp_weight)
                out_avg = out_avg.detach().cpu()
            noisy_psnr = psnr(out_full, noisy.detach().cpu(), max_val=1.0).item()

        if (i + 1) % 50:
            if noisy_psnr - psrn_noisy_last < -4 and last_net is not None:
                print("Falling back to previous checkpoint.")

                for new_param, net_param in zip(last_net, model.parameters()):
                    net_param.data.copy_(new_param.cuda())

                total_loss = total_loss * 0
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                last_net = [x.detach().cpu() for x in model.parameters()]
                psrn_noisy_last = noisy_psnr

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            out_full = model(noisy).detach().cpu()
            if H is not None:
                out_full = out_full[:, :, :H, :W]
        if out_avg is None:
            out_avg = out_full.detach().cpu()
        else:
            out_avg = out_avg * exp_weight + out_full * (1 - exp_weight)
            out_avg = out_avg.detach().cpu()

    return out_avg


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=Loader)

    noisy = io.read_image(args.input_path)
    
    out_image = main(noisy, cfg, cfg['experiment_cfg'])

    io.write_png(out_image, args.output_path)
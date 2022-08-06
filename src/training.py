import os
import pickle
import glob
import random

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
# from kornia.losses import psnr_loss as psnr
from kornia.metrics import psnr
import torchvision

from torchvision.transforms import functional as TF
from model import build_model
from icecream import ic


from tqdm import trange


from neighbor2neighbor import AugmentNoise, generate_mask_pair, generate_subimages
# from utils.denoising_utils import *

def build_loss(cfg):
    if cfg['experiment_cfg']['mse']:
        loss = torch.nn.MSELoss()
    elif cfg['experiment_cfg']['pll']:
        loss = torch.nn.PoissonNLLLoss()            
    return loss

def train(dloader, config):
    # torch.cuda.set_per_process_memory_fraction(.6)
    experiment_cfg = config['experiment_cfg']
    output_path = config['output_dir']
    
    if config['experiment_cfg']['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # model = build_model(config)
    # .type(dtype)

    # init_state_dict = model.state_dict()
    # model.type(dtype)
    # create noise generator
    # noise_adder = AugmentNoise(style=experiment_cfg['noise'])
    running_psnr_avg = 0.0
    running_ssim_avg = 0.0
    # print(config)
    print(len(dloader))
    for idx, img in enumerate(dloader):
        if 'gtandraw' in experiment_cfg['dataset'].keys() and experiment_cfg['dataset']['gtandraw']:
            noisy, img = img
            print(type(img), len(img))
            img = img.type(dtype)
            noisy = noisy.type(dtype)
        else:
            noise_adder = AugmentNoise(style=experiment_cfg['noise'])
            img, _ = img
            img = img.type(dtype)
        
            
            #noisy image
            noisy = noise_adder.add_train_noise(img).type(dtype)
        
        # with profile(activities=[ProfilerActivity.CUDA],profile_memory=True, record_shapes=True) as prof:
            # with record_function("model_inference"):

        
        results = train_helper(img, noisy, dtype, config, experiment_cfg)
        # ic(results)
        denoised, clean_psnr, psnr_list, loss_list, lpips_list, ssims_list = results
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            # print(prof.key_averages().table(sort_by="self_gpu_memory_usage", row_limit=10))
            # print(prof.key_averages().table(sort_by="gpu_memory_usage", row_limit=10))
        image_results_path = os.path.join(output_path, str(idx))
        try:
            print(image_results_path)
            os.mkdir(image_results_path)
        except:
            pass
        # write clean and noisy image
        torchvision.utils.save_image(img, os.path.join(image_results_path,"{}_gt.png".format(idx)))
        torchvision.utils.save_image(denoised, os.path.join(image_results_path,"{}_{:.3f}_out.png".format(idx, clean_psnr)))
        torchvision.utils.save_image(noisy, os.path.join(image_results_path,"{}_noisy.png".format(idx)))

        # torch.save(model)
        with open(os.path.join(image_results_path, 'metrics_psnr.pkl'), 'wb') as file:
            # yaml.dump(result_i['metrics'], yaml_file, default_flow_style=False)
            pickle.dump({'loss':loss_list, 'psnr':psnr_list, 'lpips':lpips_list, 'ssims': ssims_list}, file)
        running_psnr_avg += clean_psnr

        torch.cuda.empty_cache()
        #clean up
        # del model
        # model = build_model(config)
        # model.load_state_dict(init_state_dict)
        # model.type(dtype)

    print("#############################\n Final Average PSNR: {} SSIM:{}".format(running_psnr_avg/ len(dloader), running_ssim_avg/ len(dloader)))





def train_helper( img, noisy, dtype, config, experiment_cfg):

    return nb2nb_aug_helper( img, noisy, dtype, config, experiment_cfg)

def nb2nb_aug_helper( img, noisy, dtype, config, experiment_cfg):
    model = build_model(config)
    model.type(dtype)

    print("Number of params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # swa_model = AveragedModel(model)
    # loss 
    # loss_fn_alex = lpips.LPIPS(net='alex')
    # mse = torch.nn.MSELoss().type(dtype)
    # optimizer
    if experiment_cfg['optimizer'] == 'Adam':
        LR = experiment_cfg['lr']
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR)

    if 'lr_sched' in experiment_cfg.keys():
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.99)
    # swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    psnr_list = []
    loss_list = []
    lpips_list =[]
    ssims_list = []
    grad_hist = []
    exp_weight = .99

    out_avg = None
    # optimize single image




    noisy_in = noisy
    # WITH CSCNET did nothing somehow
    if 'rotate' in experiment_cfg.keys():
        noisy_in =  torch.cat((noisy, TF.rotate(noisy, 90), TF.rotate(noisy, 180), TF.rotate(noisy, 270)))
    
    if 'flip' in experiment_cfg.keys():
        noisy_in = torch.cat((noisy_in, TF.hflip(noisy_in), TF.vflip(noisy_in)))
    #horizonal and vertical flip
    #we need to pad to a square if the image is not already a square
    H  = None
    W = None
    if noisy.shape[2] != noisy.shape[3]:
        H = noisy.shape[2]
        W = noisy.shape[3]
        val_size = (max(H, W) + 31) // 32 * 32
        noisy_in = TF.pad(noisy, (0, 0, val_size-noisy.shape[3], val_size-noisy.shape[2]), padding_mode='reflect')
    # noisy_preaug = noisy_in
    t = trange(experiment_cfg['num_iter'])
    pll = nn.PoissonNLLLoss(log_input=False, full=True)
    last_net = None
    grad_hist = []
    # model.enable_warmup()
    # dict_checkpointC = model.apply_C.weight.data.clone().cpu()
    # dict_checkpointA = model.apply_A.weight.data.clone().cpu()
    # dict_checkpointB = model.apply_B.weight.data.clone().cpu()
    warmup  = True
    warmup_counter = 0
    psrn_noisy_last =0.0
    for i in t:

        # if i==1000 or (i > 1000 and warmup and warmup_counter == 50 ):
        #     dict_checkpointC = model.apply_C.weight.data.clone().cpu()
        #     dict_checkpointA = model.apply_A.weight.data.clone().cpu()
        #     dict_checkpointB = model.apply_B.weight.data.clone().cpu()
        #     warmup_counter = 0
        #     warmup = False
        #     # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*.1
        #     model.disable_warmup()
        #     print("Disable Warmup: {}".format(i))
        # elif i > 1000 and warmup_counter == 200 and not warmup:
        #     warmup_counter = 0
        #     warmup = True
        #     model.enable_warmup()
        #     print("Enable Warmup: {}".format(i))
        # warmup_counter += 1
        mask1, mask2 = generate_mask_pair(noisy_in)
        #g1(y) #g2(y)
        # noisy_sub1 = generate_subimages(noisy_in, mask1)
        # noisy_sub2 = generate_subimages(noisy_in, mask2)
        
        

        # if experiment_cfg['regularizer']:
        with torch.no_grad():
            # if out_avg is None:
            # if config['model_cfg']['model_type'] == 'deepcdl':
            #     noisy_denoised = model(noisy_in, torch.Tensor([50/255]).unsqueeze(1).unsqueeze(1).unsqueeze(1).type(dtype))
            # else:
            noisy_denoised = model(noisy_in)
            noisy_denoised = torch.clamp(noisy_denoised, 0.0, 1.0)
            
            # else:
                # noisy_denoised - out_avg.clone().type(dtype)    
        
        if 'cutnoise' in experiment_cfg.keys():
            noisy_denoised, noisy_in_aug = cutNoise(noisy_denoised.clone(), noisy_in.clone())
        else:
            noisy_in_aug = noisy_in.clone()
        # ic(noisy_in_aug.shape, mask1.shape, noisy_in.shape)
        noisy_sub1 = generate_subimages(noisy_in_aug, mask1)
        noisy_sub2 = generate_subimages(noisy_in_aug, mask2)


        #TODO Add noise to sub1?

        # ic(noisy_denoised.shape)
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)
        # ic(config)
        # if config['model_cfg']['model_type'] == 'deepcdl':
        #     noisy_output = model(noisy_sub1, torch.Tensor([50/255]).unsqueeze(1).unsqueeze(1).unsqueeze(1).type(dtype))
        # else:
        noisy_output = model(noisy_sub1)
        # print("MODEL: {}".format(i))
        noisy_output = torch.clamp(noisy_output, 0.0, 1.0)
        # if H is not None:
        #         noisy_output = noisy_output[:,:, :H, :W]
        noisy_target = noisy_sub2

        # ic(noisy_output.shape)
        Lambda = experiment_cfg['LAM']
        # Lambda = i /experiment_cfg['num_iter']  * experiment_cfg['LAM']
        # ic(noisy_output.shape, noisy_target.shape)
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
        # if cfg['experiment_cfg']['loss'] == 'poisson':
        #     total_loss = pll(out, img_noisy_torch)
        if "l1" in experiment_cfg.keys():
            l1_regularization = 0.
            for param in model.parameters():
                l1_regularization += param.abs().sum()
            total_loss = (experiment_cfg['l1'] * l1_regularization)
        # else:
        if 'poisson_loss' in experiment_cfg.keys():
            loss1 = pll(noisy_output, noisy_target)
            loss2 = F.l1_loss(noisy_output, noisy_target)
            loss1 += loss2
            # gamma = .5
            # loss1  = (gamma *loss1) + ((1-gamma) * loss2)
        elif 'poisson_loss_only' in experiment_cfg.keys():
            loss1 = pll(noisy_output, noisy_target)
            # loss2 = F.l1_loss(noisy_output, noisy_target)
            # loss1 += loss2
            # gamma = .5
            # loss1  = (gamma *loss1) + ((1-gamma) * loss2)
        elif 'l1_loss' in experiment_cfg.keys():
            # loss1 = pll(noisy_output, noisy_target)
            loss1 = F.l1_loss(noisy_output, noisy_target)
            # loss1 += loss2
        elif 'sure_loss' in experiment_cfg.keys():
            n = torch.randn((noisy_output.shape), requires_grad=True).type(dtype)
            div = (n@noisy_output).sum()
            div = torch.autograd.grad(div, noisy_output, retain_graph=True)[0]

            loss1 = F.l1_loss(noisy_output, noisy_target) + (n @ div).mean()

        elif 'wsure' in experiment_cfg.keys():
            # fidelity_loss = F.l1_loss(noisy_output, noisy_target)
            fidelity_loss = torch.mean(diff**2)
            epsilon = 1e-3
            eta = noisy_sub1.clone().normal_()
            net_input_perturbed = noisy_sub1.clone() + (eta * epsilon) 
            out_perturbed = model (net_input_perturbed)
            dx = out_perturbed - noisy_output
            eta_dx = torch.sum(eta * dx)
            MCdiv = eta_dx / epsilon
            div_term = 2. * (50/255) ** 2 * MCdiv / torch.numel(noisy_sub1)
            loss1 = fidelity_loss - (50/255) **2 + div_term
        elif "mse" in experiment_cfg.keys():
            loss1 = torch.mean(diff**2)
        else:
            loss1 = F.l1_loss(noisy_output, noisy_target)
            # orch.mean(diff**2)
        # loss1 = F.poisson_nll_loss(noisy_output, noisy_target, log_input=False)
        # loss1 = torch.nn.functional.l1_loss(noisy_output, noisy_target)
        loss2 = Lambda * torch.mean((diff - exp_diff)**2)

        loss = loss1 + loss2 
        if "l1" in experiment_cfg.keys():
            loss += total_loss
        loss.backward()
        
        
        
        with torch.no_grad():
                # if i > 500:
                #     out_full = swa_model(noisy).detach().cpu()
                # else:
            out_full = model(noisy).detach().cpu()
            if H is not None:
                    out_full = out_full[:,:, :H, :W]
            if out_avg is None:
                out_avg = out_full.detach().cpu()
            else:
                out_avg = out_avg * exp_weight + out_full * (1 - exp_weight)
                out_avg = out_avg.detach().cpu()
            clean_psnr = psnr(out_full, img.detach().cpu(), max_val=1.0).item()
            noisy_psnr = psnr(out_full, noisy.detach().cpu(), max_val=1.0).item()
            clean_psnr_avg = psnr(out_avg, img.detach().cpu(), max_val=1.0).item()
            
            
        if (i+1) % 50:
            if noisy_psnr - psrn_noisy_last < -4 and last_net is not None:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, model.parameters()):
                    net_param.data.copy_(new_param.cuda())

                total_loss = total_loss*0
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                last_net = [x.detach().cpu() for x in model.parameters()]
                psrn_noisy_last = noisy_psnr
        
        
        
        
        optimizer.step()
        if 'param_noise_sigma' in experiment_cfg.keys():
            add_noise(model, experiment_cfg
                  ['param_noise_sigma'], learning_rate=LR, dtype=dtype)
        # new_model_params = [p.grad.data.clone().detach().cpu() for p in model.parameters()]
        # if i > 500:
        #     swa_model.update_parameters(model)
        #     swa_scheduler.step()
        optimizer.zero_grad()

        if 'lr_sched' in experiment_cfg.keys():
            scheduler.step()


        with torch.no_grad():
            # if i > 500:
            #     out_full = swa_model(noisy).detach().cpu()
            # else:
            out_full = model(noisy).detach().cpu()
            if H is not None:
                out_full = out_full[:,:, :H, :W]
        if out_avg is None:
            out_avg = out_full.detach().cpu()
        else:
            out_avg = out_avg * exp_weight + out_full * (1 - exp_weight)
            out_avg = out_avg.detach().cpu()
        clean_psnr = psnr(out_full, img.detach().cpu(), max_val=1.0).item()
        clean_psnr_avg = psnr(out_avg, img.detach().cpu(), max_val=1.0).item()
        from skimage.metrics import structural_similarity as ssim
        # print(out_avg.shape, img.shape)
        clean_ssim = ssim(out_avg.detach().cpu().numpy().squeeze(0).squeeze(0), img.detach().cpu().numpy().squeeze(0).squeeze(0))
        ssims_list.append(clean_ssim)
        # with torch.no_grad():
            # lpips_score =  loss_fn_alex(out_avg.detach().cpu(), img.cpu()).item()
        t.set_description("PSNR:{:.5f} db | AVG:{:.5f} | | Loss: {:.5f} | SSIM: {:.5f}".format(clean_psnr, clean_psnr_avg, loss.item(), clean_ssim))
        psnr_list.append(clean_psnr)
        loss_list.append(loss.item())
        # lpips_list.append(lpips_score)
        # scheduler.step(loss)


    lpips_list = [0.0]
    clean_psnr = psnr(out_avg, img.detach().cpu(), max_val=1.0)
    # torch.save(model, '/home/cegrad/calta/sparse-dip/testmodel.pth')
    return out_avg, clean_psnr.item(), psnr_list, loss_list, lpips_list, ssims_list



def add_noise(model, param_noise_sigma, learning_rate, dtype):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size())*param_noise_sigma*learning_rate
        noise = noise.type(dtype)
        n.data = n.data + noise


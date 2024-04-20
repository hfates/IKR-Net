import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 02:01:21 2021

@author: asfand
"""
from scipy import ndimage
import os.path
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
from scipy.io import loadmat
from models.BLDSRnet_v2 import BLDSRNet_v2 as net

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def BLDSRnet(opt_P, opt_C):
    
    model = net(opt_P,opt_C,n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                       nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    #model.load_state_dict(torch.load('../checkpoints/usrnet.pth'), strict=False)
    model.load_state_dict(torch.load('../checkpoints/F_13500.pth'), strict=False)
    model.eval()
    for key, v in model.named_parameters():
        print(key)
        v.requires_grad = False
    model = model.to(device) 
    return model

#### options
parser = argparse.ArgumentParser()

parser.add_argument('-opt_P', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('-opt_C', type=str, required=True, help='Path to options YMAL file.')

opt_P = option.parse(parser.parse_args().opt_P, is_train=False)
opt_C = option.parse(parser.parse_args().opt_C, is_train=False)

# opt_F = option.dict_to_nonedict(opt_F)
opt_P = option.dict_to_nonedict(opt_P)
opt_C = option.dict_to_nonedict(opt_C)

#### mkdir and logger
util.mkdirs((path for key, path in opt_P['path'].items() if not key == 'experiments_root'
             and 'pretrain_model' not in key and 'resume' not in key))
util.mkdirs((path for key, path in opt_C['path'].items() if not key == 'experiments_root'
             and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt_P['path']['log'], 'test_' + opt_P['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
#HFA
logger.info(option.dict2str(opt_P))
logger.info(option.dict2str(opt_C))

# load PCA matrix of enough kernel
print('load PCA matrix')
pca_matrix = torch.load('./pca_matrix.pth',map_location=lambda storage, loc: storage)
print('PCA matrix shape: {}'.format(pca_matrix.shape))
enc= util.PCAEncoder(pca_matrix, cuda=False)
inv_pca = torch.transpose(enc.weight,0,1)

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt_P['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model_SR = BLDSRnet(opt_P,opt_C) 
model_SR.eval()

kernels = loadmat('kernels_12.mat')['kernels']
    
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']#path opt['']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt_P['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    for test_data in test_loader:
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        #img_path = test_data['GT_path'][0] if need_GT else test_data['LQ_path'][0]
        #img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ  
        if (0):
        #if (np.random.randint(2)):
            prepro = util.USRNetPreprocessing(opt_P['scale'], pca_matrix, random=True, para_input=opt_P['code_length'],
                                            kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                            sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=0.5, scaling=3,
                                            rate_cln=0.2, noise_high=0.0)
            LR_img, ker_map, b_kernels = prepro(test_data['GT'], kernel=True)
        else:
            filt_ind = np.random.randint(12)
            filt_ind = 5
            filt = kernels[0, filt_ind].astype(np.float64)
            filt = filt[2:-2,2:-2]
            prepro = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                                      kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                                      sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                                      rate_cln=0.2, noise_high=0.0)
            LR_img, b_kernels = prepro(test_data['GT'], filt,  kernel=True)
        
        B, H, W = 1,21,21 #[B, l, l]
         
        sigma = torch.tensor([0]*B).float().view([B, 1, 1, 1]).to(device)
        SR_it = []
        ker_it = []
        with torch.no_grad():
            SR_img, _ = model_SR(LR_img, b_kernels, opt_P['scale'], sigma, SR_it, ker_it)    

        #sr_img = util.tensor2img(SR_img.detach().cpu())  # uint8
        
        ###########################
        # Save images for reference
        step = 1
        lr_img = util.tensor2img(LR_img) #save LR image for reference
        img_name = os.path.splitext(os.path.basename(test_data['LQ_path'][0]))[0]
        img_dir = os.path.join(dataset_dir, img_name)
        # img_dir = os.path.join(opt_F['path']['val_images'], str(current_step), '_', str(step))       
        util.mkdir(img_dir)
        save_lr_path = os.path.join(img_dir, '{:s}_LR.png'.format(img_name))
        util.save_img(lr_img, save_lr_path)
        #SR_img= torch.nn.functional.interpolate(LR_img, scale_factor=opt_P['scale'], mode='nearest')
        for indSR in range(8): 
            sr_img = util.tensor2img(SR_it[indSR])
            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, indSR))
            util.save_img(sr_img, save_img_path)

            ker_cur = ker_it[indSR]
            ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
            save_ker_path = os.path.join(img_dir, 'ker_{:d}.png'.format(indSR))
            util.save_img(ker_img, save_ker_path)
        
        ker_cur = b_kernels.detach().cpu()
        ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
        save_ker_path = os.path.join(img_dir, 'ker_org.png')
        util.save_img(ker_img, save_ker_path)
        
        # calculate PSNR and SSIM    
        if need_GT:
            gt_img = util.tensor2img(test_data['GT'].detach().cpu())  # uint8
            
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.

            crop_border = opt_P['crop_border'] if opt_P['crop_border'] else opt_P['scale']
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                if crop_border == 0:
                    cropped_sr_img_y = sr_img_y
                    cropped_gt_img_y = gt_img_y
                else:
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                single_img_psnr.append( psnr)
                single_img_ssim.append( ssim)
                single_img_psnr_y.append(psnr_y)
                single_img_ssim_y .append( ssim_y)
                logger.info(
                    'step:{:3d}, img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                        format(step, img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                logger.info('step:{:3d}, img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(step, img_name, psnr, ssim))
        else:
            logger.info(img_name)
            
        if need_GT:
            max_img_index = 0 #np.argmax(single_img_psnr)
            test_results['psnr'].append(single_img_psnr[max_img_index])
            test_results['ssim'].append(single_img_ssim[max_img_index])
            test_results['psnr_y'].append(single_img_psnr_y[max_img_index])
            test_results['ssim_y'].append(single_img_ssim_y[max_img_index])
            
            avg_signle_img_psnr = sum(single_img_psnr) / step
            avg_signle_img_ssim = sum(single_img_ssim )/ step
            avg_signle_img_psnr_y = sum(single_img_psnr_y) / step
            avg_signle_img_ssim_y = sum(single_img_ssim_y) / step
            # logger.info(
            #     'step:{:3d}, img:{:15s} - average PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
            #         format(step, img_name, avg_signle_img_psnr, avg_signle_img_ssim, avg_signle_img_psnr_y, avg_signle_img_ssim_y))
            # max_signle_img_psnr = single_img_psnr[max_img_index]
            # max_signle_img_ssim = single_img_ssim[max_img_index]
            # max_signle_img_psnr_y = single_img_psnr_y[max_img_index]
            # max_signle_img_ssim_y = single_img_ssim_y[max_img_index]
            # logger.info(
            #     'step:{:3d}, img:{:15s} - max PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
            #         format(step, img_name, max_signle_img_psnr, max_signle_img_ssim, max_signle_img_psnr_y, max_signle_img_ssim_y))
      
        
    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info(
            '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                test_set_name, ave_psnr, ave_ssim))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info(
                '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                format(ave_psnr_y, ave_ssim_y))


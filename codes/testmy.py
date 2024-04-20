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
from models.usrnet import USRNet as net
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def USRNET():
    
    model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                       nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    model.load_state_dict(torch.load('../checkpoints/usrnet.pth'), strict=True)
    model.eval()
    for key, v in model.named_parameters():
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
logger.info(option.dict2str(opt_P))
logger.info(option.dict2str(opt_C))
# load PCA matrix of enough kernel
print('load PCA matrix')
pca_matrix = torch.load('./pca_matrix.pth',map_location=lambda storage, loc: storage)
print('PCA matrix shape: {}'.format(pca_matrix.shape))
#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt_P['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model_F = USRNET()
model_P = create_model(opt_P)
model_C = create_model(opt_C)
enc= util.PCAEncoder(pca_matrix, cuda=True)
kernels = loadmat('kernels_12.mat')['kernels']

   
kernel = kernels[0, 0].astype(np.float64)
    
inv_pca = torch.transpose(enc.weight,0,1)
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
        img_path = test_data['GT_path'][0] if need_GT else test_data['LQ_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        # prepro = util.USRNetPreprocessing(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
        #                                           kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
        #                                           sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
        #                                           rate_cln=0.2, noise_high=0.0)
        #     #HFA
        # LR_img, ker_map, b_kernels = prepro(test_data['GT'], kernel=True)
        #LR_img = ndimage.filters.convolve(test_data['GT'][0], kernel[..., np.newaxis], mode='wrap')  # blur
        LR_img = ndimage.filters.convolve(test_data['GT'][0].permute(1,2,0), kernel[..., np.newaxis], mode='wrap')  # blur
        LR_img=LR_img[::opt_P['scale'], ::opt_P['scale'], ...]
         # downsample, standard s-fold downsampler
        
        LR_img=torch.from_numpy(np.ascontiguousarray(LR_img)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        B, H, W =(1,21,21)
        
        # LR_img = test_data['LQ'].to(device)

        # Predictor test
        model_P.feed_data(LR_img)
        model_P.test()
        P_visuals = model_P.get_current_visuals()
        est_ker_map = P_visuals['Batch_est_ker_map']
        # save LR image
        save_img_path = os.path.join(dataset_dir, img_name + '_LR.png')
        util.save_img(util.tensor2img(LR_img.detach().cpu()), save_img_path)
        
       
        # Corrector test
       
        for step in range(opt_C['step']):
            step += 1
            
            b_check = torch.bmm( est_ker_map.to(device).expand((B, )+est_ker_map.size()), inv_pca.expand((B, ) + inv_pca.size())).view((B, -1))
            b_check = torch.add(b_check[0], 1/(H*W)).view((B, H , W)) 
            b_check = b_check.view(B,1,H,W)
            # est_ker_map=util.single2tensor4(est_ker_map[..., np.newaxis]).to(device)
            # Test USRNET to produce SR images
            sigma = torch.tensor(0).float().view([1, 1, 1, 1]).to(device)
            SR_img=model_F(LR_img , b_check,opt_P['scale'],sigma)
            
            model_C.feed_data(SR_img, est_ker_map)
            model_C.test()
            C_visuals = model_C.get_current_visuals()
            est_ker_map = C_visuals['Batch_est_ker_map']
          
            # save images
            sr_img = util.tensor2img(SR_img.detach().cpu())  # uint8
            suffix = opt_P['suffix']
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + '_' + str(step) + '.png')
            else:
                save_img_path = os.path.join(dataset_dir, img_name + '_' + str(step) + '.png')
            util.save_img(sr_img, save_img_path)
            
            # calculate PSNR and SSIM
            if need_GT:
                
                #gt_img = util.tensor2img(torch.from_numpy(np.ascontiguousarray(test_data['GT'][0])).permute(2, 0, 1).float().unsqueeze(0).detach().cpu())
                gt_img = util.tensor2img(torch.from_numpy(np.ascontiguousarray(test_data['GT'][0])).float().unsqueeze(0).detach().cpu())
 
                # gt2 = gt_img[100:300,200:400,:]
                # save_img_path = os.path.join(dataset_dir, img_name + '_'  + 'gt.png')
                # util.save_img(gt2, save_img_path)
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
            max_img_index = 6 # np.argmax(single_img_psnr)
            test_results['psnr'].append(single_img_psnr[max_img_index])
            test_results['ssim'].append(single_img_ssim[max_img_index])
            test_results['psnr_y'].append(single_img_psnr_y[max_img_index])
            test_results['ssim_y'].append(single_img_ssim_y[max_img_index])
            
            avg_signle_img_psnr = sum(single_img_psnr) / step
            avg_signle_img_ssim = sum(single_img_ssim )/ step
            avg_signle_img_psnr_y = sum(single_img_psnr_y) / step
            avg_signle_img_ssim_y = sum(single_img_ssim_y) / step
            logger.info(
                'step:{:3d}, img:{:15s} - average PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                    format(step, img_name, avg_signle_img_psnr, avg_signle_img_ssim, avg_signle_img_psnr_y, avg_signle_img_ssim_y))
            max_signle_img_psnr = single_img_psnr[max_img_index]
            max_signle_img_ssim = single_img_ssim[max_img_index]
            max_signle_img_psnr_y = single_img_psnr_y[max_img_index]
            max_signle_img_ssim_y = single_img_ssim_y[max_img_index]
            logger.info(
                'step:{:3d}, img:{:15s} - max PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                    format(step, img_name, max_signle_img_psnr, max_signle_img_ssim, max_signle_img_psnr_y, max_signle_img_ssim_y))
      
        
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


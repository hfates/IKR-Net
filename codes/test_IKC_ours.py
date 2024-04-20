import os.path
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
import options.options as option
#import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils import util
from scipy.io import loadmat
import cv2
#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt_F', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('-opt_P', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('-opt_C', type=str, required=True, help='Path to options YMAL file.')
opt_F = option.parse(parser.parse_args().opt_F, is_train=False)
opt_P = option.parse(parser.parse_args().opt_P, is_train=False)
opt_C = option.parse(parser.parse_args().opt_C, is_train=False)

opt_F = option.dict_to_nonedict(opt_F)
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

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt_P['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model_F = create_model(opt_F)
model_P = create_model(opt_P)
model_C = create_model(opt_C)

# load PCA matrix of enough kernel
print('load PCA matrix')
pca_matrix = torch.load('./pca_matrix.pth',map_location=lambda storage, loc: storage)
print('PCA matrix shape: {}'.format(pca_matrix.shape))
#Kernel Generation
kernels = loadmat('kernels_12.mat')['kernels']
motion_kernels=torch.load('../data_samples/blur_kernels/motion.m')
for filt_ind in range(8):
    kernels[0, filt_ind] = kernels[0, filt_ind][2:-2,2:-2]
for filt_ind in range(4):
    kernels[0, filt_ind+8] = np.array(motion_kernels[filt_ind], dtype=np.float64) 

filt_ind = 0
noise_level = 0.0

#for filt_ind in range(12):
for filt_ind in [3]:
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']#path opt['']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = os.path.join(opt_P['path']['results_root'], test_set_name+str(filt_ind))
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
            filt = kernels[0, filt_ind].astype(np.float64)                       
            prepro_val = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                                      kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                                      sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                                      rate_cln=0.2, noise_high=noise_level)
            LR_img, b_kernels, noise_level_rand = prepro_val(test_data['GT'], filt,  kernel=True)
            
            #prepro = util.SRMDPreprocessing(opt_F['scale'], pca_matrix, random=False, para_input=opt_F['code_length'], noise=False, cuda=True,
            #                                 sig=opt_F['sig'], sig_min=opt_F['sig_min'], sig_max=opt_F['sig_max'], rate_iso=1.0, scaling=3,
            #                                 rate_cln=0.2, noise_high=0.0) # random(sig_min, sig_max) | stable kernel(sig)
            # LR_img, ker_map = prepro(test_data['GT'])
            #LR_img = test_data['LQ']
    
            # Predictor test
            model_P.feed_data(LR_img)
            model_P.test()
            P_visuals = model_P.get_current_visuals()
            est_ker_map = P_visuals['Batch_est_ker_map']
    
            # Corrector test
            for step in range(opt_C['step']):
                step += 1
                # Test SFTMD to produce SR images
                model_F.feed_data(test_data, LR_img, est_ker_map)
                model_F.test()
                F_visuals = model_F.get_current_visuals()
                SR_img = F_visuals['Batch_SR']
    
                model_C.feed_data(SR_img, est_ker_map)
                model_C.test()
                C_visuals = model_C.get_current_visuals()
                est_ker_map = C_visuals['Batch_est_ker_map']
    
                sr_img = util.tensor2img(F_visuals['SR'])  # uint8
                
                srnew = cv2.resize(sr_img, (sr_img.shape[1]*2, sr_img.shape[0]*2), cv2.INTER_CUBIC)
                #sr_img = srnew[0::2,0::2,:] 
                sr_img = 0.5*srnew[1:-1:2,1:-1:2,:] + 0.5*srnew[2::2,2::2,:] 
                #srnew = cv2.resize(sr_img, (sr_img.shape[1]*4, sr_img.shape[0]*4), cv2.INTER_CUBIC)
                # #sr_img = srnew[3::4,3::4,:]
                # sr_img = 0.5*srnew[3:-1:4,3:-1:4,:] + 0.5*srnew[4::4,4::4,:]
                shift_no = 0
                
                # save images
                suffix = opt_P['suffix']
                if (step==opt_C['step']-1):
                    if suffix:
                        save_img_path = os.path.join(dataset_dir, img_name + suffix + '_' + str(step) + '.png')
                    else:
                        save_img_path = os.path.join(dataset_dir, img_name + '_' + str(step) + '.png')
                    util.save_img(sr_img, save_img_path)
    
                # calculate PSNR and SSIM
                if need_GT:
                    gt_img = util.tensor2img(F_visuals['GT'])
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
    
                    crop_border = opt_P['crop_border'] if opt_P['crop_border'] else opt_P['scale']
                    if crop_border == 0:
                        cropped_sr_img = sr_img
                        cropped_gt_img = gt_img
                    else:
                        #cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_sr_img = sr_img[crop_border+shift_no:-crop_border+shift_no+1, crop_border+shift_no:-crop_border+shift_no+1, :]
                        cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
    
                    psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                    #test_results['psnr'].append(psnr)
                    #test_results['ssim'].append(ssim)
    
                    if gt_img.shape[2] == 3:  # RGB image
                        sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                        if crop_border == 0:
                            cropped_sr_img_y = sr_img_y
                            cropped_gt_img_y = gt_img_y
                        else:
                            #cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                            cropped_sr_img_y = sr_img_y[crop_border+shift_no:-crop_border+shift_no+1, crop_border+shift_no:-crop_border+shift_no+1]
                            cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                        psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                        ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                        #test_results['psnr_y'].append(psnr_y)
                        #test_results['ssim_y'].append(ssim_y)
                        single_img_psnr += [psnr]
                        single_img_ssim += [ssim]
                        single_img_psnr_y += [psnr_y]
                        single_img_ssim_y += [ssim_y]
                        logger.info(
                            'step:{:3d}, img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                                format(step, img_name, psnr, ssim, psnr_y, ssim_y))
                    else:
                        logger.info('step:{:3d}, img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(step, img_name, psnr, ssim))
                else:
                    logger.info(img_name)
    
            if need_GT:
                #max_img_index = np.argmax(single_img_psnr)
                max_img_index = opt_C['step']-1
                test_results['psnr'].append(single_img_psnr[max_img_index])
                test_results['ssim'].append(single_img_ssim[max_img_index])
                test_results['psnr_y'].append(single_img_psnr_y[max_img_index])
                test_results['ssim_y'].append(single_img_ssim_y[max_img_index])
                
                avg_signle_img_psnr = sum(single_img_psnr) / step
                avg_signle_img_ssim = sum(single_img_ssim) / step
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


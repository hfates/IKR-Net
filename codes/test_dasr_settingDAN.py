import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from IPython import embed

import DAN_test.options as option

sys.path.insert(0, "../../")
import DAN_test.utils as util
from DAN_test.data import create_dataloader, create_dataset
from DAN_test.data.util import bgr2ycbcr

import cv2
####
from models.BLDSRnet_v2 import BLDKerNet as net
from torch.nn.parallel import DataParallel
from models import create_model
import options.options as option_dasr
from scipy.io import loadmat

sys.path.append('/home/hfates/Documents/deep_projects/DeepImage/DASR/')
import utility
from argparse import Namespace

model_no = 88
sc_final = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    #### setup options of three networks
    args = Namespace(batch_size=32, beta1=0.9, beta2=0.999, blur_kernel=21, blur_type='aniso_gaussian', noise=0.0,scale=[4.0],
                 chop=False, cpu=False, data_range='1-3450/801-810', data_test='B100', data_train='DF2K',
                 debug=False, decay_type='step', dilation=False, dir_data='/home/fbyaman/Downloads/benchmark', dir_demo='../test',
                 epochs_encoder=100, epochs_sr=500, epsilon=1e-08, ext='sep', extend='.', gamma_encoder=0.1,
                 gamma_sr=0.5, lambda_1=0.2, lambda_2=4.0, lambda_max=4.0, lambda_min=0.2, load='.', loss='1*L1',
                 lr_decay_encoder=60, lr_decay_sr=125, lr_encoder=0.001, lr_sr=0.0001, mode='bicubic', model='blindsr',
                 momentum=0.9, n_GPUs=1, n_colors=3, n_threads=4, no_augment=False,  optimizer='ADAM',
                 patch_size=48, pre_train='.', precision='single', print_every=200, reset=False, resume=600,
                 rgb_range=255, save='blindsr', save_models=False, save_results=False,  seed=1,
                 self_ensemble=False, shift_mean=True, sig=4.0, sig_max=4.0, sig_min=0.2, skip_threshold=1000000.0,
                 split_batch=1, start_epoch=0, template='.', test_every=1000, test_only=True, theta=0.0, weight_decay=0)
    os.chdir('/home/hfates/Documents/deep_projects/DeepImage/DASR/')
    import model
    from model.blindsr import BlindSR
    
    # torch.manual_seed(args.seed)
    #checkpoint = utility.checkpoint(args)
    #model_DSR = model.Model(args, checkpoint)
    if args.blur_type == 'iso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_iso'
    elif args.blur_type == 'aniso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso' 
        
    DASR = BlindSR(args).cuda()
    #DASR.load_state_dict(torch.load(dir + '/model/model_' + str(args.resume) + '.pt'), strict=False)
    DASR.load_state_dict(torch.load(dir + '/model/model_' + str(model_no) + '.pt'), strict=False)
    DASR.eval()
    
    os.chdir('/home/hfates/Documents/deep_projects/DeepImage/BLDSR/codes')
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_P', type=str, default="options/test/test_DASR_p.yml", help='Path to option YMAL file of Predictor.')
    parser.add_argument('-opt_C', type=str, default="options/test/test_DASR_c.yml", help='Path to option YMAL file of Corrector.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt_P = option_dasr.parse(args.opt_P, is_train=True)
    opt_C = option_dasr.parse(args.opt_C, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt_P = option_dasr.dict_to_nonedict(opt_P)
    opt_C = option_dasr.dict_to_nonedict(opt_C)
    

    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        "../../DAN/pca_matrix/DANv2/pca_matrix.pth", map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))
    
    up_scale = 4
    mod_scale = 4
    degradation_setting = {
        "random_kernel": False,
        "code_length": 10,
        "ksize": 21,
        "pca_matrix": pca_matrix,
        "scale": up_scale,
        "cuda": True,
        "rate_iso": 1.0
    }
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default='options/setting2/test/test_setting2_x4_HFA.yml', required=False, help="Path to options YMAL file.")
    opt = option.parse(parser.parse_args().opt, is_train=False)
    
    opt = option.dict_to_nonedict(opt)
    
    #### mkdir and logger
    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if not key == "experiments_root"
            and "pretrain_model" not in key
            and "resume" not in key
        )
    )
    
    os.system("rm ./result")
    os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")
    
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))
    
    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info(
            "Number of test images in [{:s}]: {:d}".format(
                dataset_opt["name"], len(test_set)
            )
        )
        test_loaders.append(test_loader)
    

    # load pretrained model by default
    #model = create_model(opt)
    #model_SR = BLDSRnet(opt_P,opt_C)  #load pretrained model of USRNET
    
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]  # path opt['']
        logger.info("\nTesting [{:s}]...".format(test_set_name))
        test_start_time = time.time()
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
        util.mkdir(dataset_dir)
    
        test_results = OrderedDict()
        test_results["psnr"] = []
        test_results["ssim"] = []
        test_results["psnr_y"] = []
        test_results["ssim_y"] = []
    
        for i,test_data in enumerate(test_loader):
            # if (i>=68):
            #     break
            single_img_psnr = []
            single_img_ssim = []
            single_img_psnr_y = []
            single_img_ssim_y = []
            need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
            img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
            img_name = img_path
            # img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # sig = 3.0
            # prepro = util.SRMDPreprocessing(sig=sig, **degradation_setting)
            # img_HR = test_data["GT"].to(device)
            # LR_img, ker_map, b_kernels = prepro(img_HR, kernel=True)    
            
            #### input dataset_LQ
            # model.feed_data(test_data["LQ"], test_data["GT"])
            # model.test()
            # visuals = model.get_current_visuals()
            # SR_img = visuals["Batch_SR"]
            # sr_img = util.tensor2img(visuals["SR"].squeeze())  # uint8
            
            b_kernels = torch.zeros(1,21,21).to(device)
            sigma = torch.tensor(0).float().view([1, 1, 1, 1]).to(device)
            with torch.no_grad():
                SR_img = DASR(test_data["LQ"].to(device) * 255)
                SR_img /= 255.0
                
            sr_img = util.tensor2img(SR_img.detach().float().cpu())  # uint8
            
            # srnew = cv2.resize(sr_img, (sr_img.shape[1]*2, sr_img.shape[0]*2), cv2.INTER_CUBIC)
            # sr_img = 0.5*srnew[1:-1:2,1:-1:2,:] + 0.5*srnew[2::2,2::2,:] 
            # #srnew = cv2.resize(sr_img, (sr_img.shape[1]*4, sr_img.shape[0]*4), cv2.INTER_CUBIC)
            # # #sr_img = srnew[3::4,3::4,:]
            # # sr_img = 0.5*srnew[3:-1:4,3:-1:4,:] + 0.5*srnew[4::4,4::4,:]
            # shift_no = -2
    
            suffix = opt["suffix"]
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
            else:
                save_img_path = os.path.join(dataset_dir, img_name + ".png")
            util.save_img(sr_img, save_img_path)
    
            if need_GT:
                gt_img = util.tensor2img(test_data["GT"].detach().float().cpu())
                gt_img = gt_img / 255.0
                sr_img = sr_img / 255.0
    
                crop_border = opt["crop_border"] if opt["crop_border"] else (opt["scale"]**2)
                if crop_border == 0:
                    cropped_sr_img = sr_img
                    cropped_gt_img = gt_img
                else:
                    cropped_sr_img = sr_img[
                        crop_border:-crop_border, crop_border:-crop_border
                    ]
                    #cropped_sr_img = sr_img[crop_border+shift_no:-crop_border+shift_no+1, crop_border+shift_no:-crop_border+shift_no+1, :]
    
                    cropped_gt_img = gt_img[
                        crop_border:-crop_border, crop_border:-crop_border
                    ]
    
                psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
    
                test_results["psnr"].append(psnr)
                test_results["ssim"].append(ssim)
    
                if len(gt_img.shape) == 3:
                    if gt_img.shape[2] == 3:  # RGB image
                        sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                        if crop_border == 0:
                            cropped_sr_img_y = sr_img_y
                            cropped_gt_img_y = gt_img_y
                        else:
                            cropped_sr_img_y = sr_img_y[
                                crop_border:-crop_border, crop_border:-crop_border
                            ]
                            #cropped_sr_img_y = sr_img_y[crop_border+shift_no:-crop_border+shift_no+1, crop_border+shift_no:-crop_border+shift_no+1]
                            cropped_gt_img_y = gt_img_y[
                                crop_border:-crop_border, crop_border:-crop_border
                            ]
                        psnr_y = util.calculate_psnr(
                            cropped_sr_img_y * 255, cropped_gt_img_y * 255
                        )
                        ssim_y = util.calculate_ssim(
                            cropped_sr_img_y * 255, cropped_gt_img_y * 255
                        )
    
                        test_results["psnr_y"].append(psnr_y)
                        test_results["ssim_y"].append(ssim_y)
    
                        logger.info(
                            "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                                img_name, psnr, ssim, psnr_y, ssim_y
                            )
                        )
                else:
                    logger.info(
                        "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                            img_name, psnr, ssim
                        )
                    )
    
                    test_results["psnr_y"].append(psnr)
                    test_results["ssim_y"].append(ssim)
            else:
                logger.info(img_name)
    
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        logger.info(
            "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
                test_set_name, ave_psnr, ave_ssim
            )
        )
        if test_results["psnr_y"] and test_results["ssim_y"]:
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
            logger.info(
                "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                    ave_psnr_y, ave_ssim_y
                )
            )
            
if __name__ == '__main__':
    main()
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
import options.options as option_ikc
from scipy.io import loadmat

sys.path.append('/home/hfates/Documents/deep_projects/DeepImage/DASR/')
import utility
from argparse import Namespace

model_no = 88
sc_final = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt_F', type=str, default="options/test/test_SFTMD.yml", help='Path to options YMAL file.')
    parser.add_argument('-opt_P', type=str, default="options/test/test_Predictor.yml", help='Path to options YMAL file.')
    parser.add_argument('-opt_C', type=str, default="options/test/test_Corrector.yml", help='Path to options YMAL file.')
    opt_F = option_ikc.parse(parser.parse_args().opt_F, is_train=False)
    opt_P = option_ikc.parse(parser.parse_args().opt_P, is_train=False)
    opt_C = option_ikc.parse(parser.parse_args().opt_C, is_train=False)
    
    opt_F = option_ikc.dict_to_nonedict(opt_F)
    opt_P = option_ikc.dict_to_nonedict(opt_P)
    opt_C = option_ikc.dict_to_nonedict(opt_C)
    args = parser.parse_args()
    # load pretrained model by default
    model_F = create_model(opt_F)
    model_P = create_model(opt_P)
    model_C = create_model(opt_C)

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
            
            # Predictor test
            LR_img = test_data["LQ"].to(device)
            model_P.feed_data(LR_img)
            model_P.test()
            P_visuals = model_P.get_current_visuals()
            est_ker_map = P_visuals['Batch_est_ker_map']
            
            # Corrector test
            for step_C in range(opt_C['step']):
                #step_C += 1
                # Test SFTMD to produce SR images
                model_F.feed_data(test_data, LR_img, est_ker_map)
                model_F.test()
                F_visuals = model_F.get_current_visuals()
                SR_img = F_visuals['Batch_SR']
    
                model_C.feed_data(SR_img, est_ker_map)
                model_C.test()
                C_visuals = model_C.get_current_visuals()
                est_ker_map = C_visuals['Batch_est_ker_map']
                
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
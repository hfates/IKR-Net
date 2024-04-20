import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import os
import math
import argparse
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchsummary import summary
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from utils import utils_image_v2 as util_image
from utils import utils_option as option_GAN
from data import create_dataloader, create_dataset
from models import create_model
from models.BLDSRnet_v2 import BLDKerNet as net
from models.model_gan import ModelGAN as M
from torch.optim import Adam
from torch.nn.parallel import DataParallel

from utils.utils_dist import get_dist_info
from utils import utils_logger

import matplotlib.pyplot as plt
from scipy.io import loadmat

from motionblur.motionblur import Kernel 

from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def BLDSRnet(opt_P, opt_C):
    
    model = net(opt_P,opt_C,n_iter=8, h_nc=64, in_nc=1, out_nc=1, nc=[2, 4, 8, 16],
                       nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    #model.load_state_dict(torch.load('../checkpoints/F_13500.pth'), strict=False)
    #model.load_state_dict(torch.load('../checkpoints/F_1000.pth'), strict=False)
    #model.load_state_dict(torch.load('../checkpoints/usrnet.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/full_usr_BSD68_ALL/save_models/USRNet_fine_tuned/F_160000.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/full_usr_BSD68_ALL/save_models/USRNet+est_fine_tuned_after100K/F_103000.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/full_usr_ALL_scale2/full_models/F_191000.pth'), strict=True)
    #model.load_state_dict(torch.load('../experiments/full_usr_ALL_noise01/full_models/F_68000.pth'), strict=True)
    #model.load_state_dict(torch.load('../experiments/results_bld_ker/HR_img_exp/models/F_180000.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/results_bld_ker/F_4000.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/results_bld_ker/SR_img_exp/F4000/F_4000.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/full_usr_BSD68_ALL/save_models/USRNet_fine_tuned/F_160000.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/full_usr_BSD68_ALL/save_models/USRNet+est_fine_tuned_after100K/F_103000.pth'), strict=False)
    #model.load_state_dict(torch.load('../checkpoints/usrnet.pth'), strict=False)
    # model.load_state_dict(torch.load('../experiments/results_bld_ker/models/F_42000.pth'), strict=True)
    # model.load_state_dict(torch.load('../experiments/train_bldsr_init_1it/models/F_22000.pth'), strict=False)
    
    # pretrained_dict = torch.load('../experiments/results_bld_ker/models/F_42000.pth')
    # pretrained_dict = {k[:]: v for (k, v) in pretrained_dict.items() if k[:2]=='pk'}
    # model.load_state_dict(pretrained_dict, strict=False)
    
    # pretrained_dict = torch.load('../experiments/full_usr_BSD68_ALL/save_models/USRNet_fine_tuned/F_160000.pth')
    # pretrained_dict_init = {k[0]+"_init"+k[1:]: v for (k, v) in pretrained_dict.items() if ((k[0]=='p') or (k[0]=='h'))}
    # pretrained_dict_else = {k : v for (k, v) in pretrained_dict.items() if ((k[0]!='p') and (k[0]!='h'))}
    # pretrained_dict = dict(pretrained_dict_init, **pretrained_dict_else)
    # model.load_state_dict(pretrained_dict, strict=False)

    # model.load_state_dict(torch.load('../checkpoints/usrnet.pth'), strict=False)
    #model.load_state_dict(torch.load('../experiments/train_8it_bld_ker/1it_bldker_models/F_11000.pth'), strict=True)

    model.p = DataParallel(model.p)
    model.h = DataParallel(model.h)
    model.p_init = DataParallel(model.p_init)
    model.h_init = DataParallel(model.h_init)
    model.load_state_dict(torch.load('../experiments/train_USRit_bld_ker_USRfine/F_3000.pth'), strict=True)

    # pretrained_dict = torch.load('../checkpoints/usrnet.pth')
    # pretrained_dict = {k[0]+"_init"+'.module'+k[1:]: v for (k, v) in pretrained_dict.items() if ((k[0]=='p') or (k[0]=='h'))}
    # model.load_state_dict(pretrained_dict, strict=False)


    #model.eval()
    for key, v in model.named_parameters():
        print(key)
        #if (key[0:4]=='netC') | (key[0:4]=='netP'):
        if (key[0]=='p') | (key[0]=='h') | (key[0:4]=='netC'):
        #if (key[0:4]=='netC'):
            v.requires_grad = False
        else:
            v.requires_grad = False
    model = model.to(device) 
    return model

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn': #Return the name of start method used for starting processes
        mp.set_start_method('spawn', force=True) ##'spawn' is the default on Windows
    rank = int(os.environ['RANK']) #system env process ranks
    num_gpus = torch.cuda.device_count() #Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs) #Initializes the default distributed process group

def prepare_opt(json_path):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option_GAN.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option_GAN.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_D, init_path_D = option_GAN.find_last_checkpoint(opt['path']['models'], net_type='D')
    init_iter_E, init_path_E = option_GAN.find_last_checkpoint(opt['path']['models'], net_type='E')
    #opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netD'] = init_path_D
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option_GAN.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    init_iter_optimizerD, init_path_optimizerD = option_GAN.find_last_checkpoint(opt['path']['models'], net_type='optimizerD')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    opt['path']['pretrained_optimizerD'] = init_path_optimizerD
    current_step = max(init_iter_G, init_iter_D, init_iter_E, init_iter_optimizerG, init_iter_optimizerD)

    # opt['path']['pretrained_netG'] = ''
    # current_step = 0
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option_GAN.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option_GAN.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option_GAN.dict2str(opt))
    return opt

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_P', type=str, default="options/train/train_Pred_EncDec.yml", help='Path to option YMAL file of Predictor.')
    parser.add_argument('-opt_C', type=str, default="options/train/train_Corr_EncDec.yml", help='Path to option YMAL file of Corrector.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt_P = option.parse(args.opt_P, is_train=True)
    opt_C = option.parse(args.opt_C, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt_P = option.dict_to_nonedict(opt_P)
    opt_C = option.dict_to_nonedict(opt_C)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #opt_F = opt_F['sftmd']
    opt_GAN = prepare_opt(json_path='options/train_bsrgan_x4_gan.json')
    
    #### set random seed
    seed = opt_P['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    # load PCA matrix of enough kernel
    print('load PCA matrix')
    pca_matrix = torch.load('./pca_matrix.pth',map_location=lambda storage, loc: storage)
    print('PCA matrix shape: {}'.format(pca_matrix.shape))

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt_P['dist'] = False
        opt_C['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt_P['dist'] = True
        opt_C['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size() #Returns the number of processes in the current process group
        rank = torch.distributed.get_rank() #Returns the rank of current process group

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt_P['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt_P['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt_P, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt_P['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt_P['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))
            # Corrector path
            util.mkdir_and_rename(
                opt_C['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt_C['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt_P['path']['log'], 'train_' + opt_P['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt_P['path']['log'], 'val_' + opt_P['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        #logger.info(option.dict2str(opt_P))
        #logger.info(option.dict2str(opt_C))
        # tensorboard logger
        if opt_P['use_tb_logger'] and 'debug' not in opt_P['name']:
            version = float(torch.__version__[0:3])
            #HFA
            if 0: #version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt_P['name'])
    else:
        util.setup_logger('base', opt_P['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')


    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt_P['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt_P['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt_P['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt_P, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt_P, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model_SR = BLDSRnet(opt_P,opt_C)  #load pretrained model of USRNET
    enc= util.PCAEncoder(pca_matrix, cuda=True)
    inv_pca = torch.transpose(enc.weight,0,1)
    
    model_GAN = M(opt_GAN, opt_P, opt_C)
    model_GAN.netG.load_state_dict(torch.load('../experiments/GAN_init_ker/models/G_143000.pth'), strict=True)
    
    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model_SR.pre.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    #####
    #Kernel Generation
    kernels = loadmat('kernels_12.mat')['kernels']
    motion_kernels=torch.load('../data_samples/blur_kernels/motion.m')
    for filt_ind in range(8):
        kernels[0, filt_ind] = kernels[0, filt_ind][2:-2,2:-2]
    for filt_ind in range(4):
        kernels[0, filt_ind+8] = np.array(motion_kernels[filt_ind], dtype=np.float64) 
        
    noise_level = 0.0
    filt_no =  100
    ker_Arr = torch.zeros([filt_no,21,21], dtype=torch.float32).to(device)
    for ind in range(filt_no):
        #if (1):
        if (np.random.randint(5)>0): # Initialise Kernel
               kernel = Kernel(size=(21, 21), intensity=np.random.rand())
               # Display kernel
               #kernel.displayKernel()
               # Get kernel as numpy array
               filt = kernel.kernelMatrix     
        else:
               filt_ind = np.random.randint(12)
               filt = kernels[0, filt_ind].astype(np.float64)
        ker_Arr[ind,:,:] = torch.FloatTensor(filt).to(device)

    #### training 
    ind_it = 0
    idxAll = np.random.permutation(filt_no)
    crop_size = opt_P['scale']**2
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(1):
        
        for filt_ind in range(12):
        #for filt_ind in [11]:
            current_step += 1

            # validation, to produce ker_map_list(fake)
            if (1):
                avg_psnr = 0.0
                avg_mse  = 0.0
                img_psnr = 0.0
                img_ssim = 0.0
                idx = 0
                model_SR.eval()
                for _, val_data in enumerate(val_loader):
                    #if (idx>10):
                    #    break
                    #if not('img_098' in val_data['LQ_path'][0]):
                    #    continue
                    # if (0):
                    # #if (np.random.randint(2)):
                    #     prepro_val = util.USRNetPreprocessing(opt_P['scale'], pca_matrix, random=True, para_input=opt_P['code_length'],
                    #                                     kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                    #                                     sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=0.5, scaling=3,
                    #                                     rate_cln=0.2, noise_high=0.0)
                    #     LR_img, ker_map, b_kernels = prepro_val(val_data['GT'], kernel=True)
                    # else:
                    #     if (np.random.randint(2)): # Initialise Kernel
                    #         kernel = Kernel(size=(21, 21), intensity=np.random.rand())
                    #         # Display kernel
                    #         #kernel.displayKernel()
                    #         # Get kernel as numpy array
                    #         filt = kernel.kernelMatrix     
                    #     else:
                            
                    #filt_ind = np.random.randint(12)
                    #filt_ind = 5 
                    filt = kernels[0, filt_ind].astype(np.float64)
                    #kernel = Kernel(size=(21, 21), intensity=np.random.rand())
                    #filt = kernel.kernelMatrix                            
                        
                    prepro_val = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                                              kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                                              sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                                              rate_cln=0.2, noise_high=noise_level)
                    LR_img, b_kernels = prepro_val(val_data['GT'], filt,  kernel=True)
                
                    #single_img_psnr = 0.0
                    lr_img = util.tensor2img(LR_img) #save LR image for reference
                    B, H, W = 1,21,21 #[B, l, l]
                                       
                    # Save images for reference
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    #img_dir = os.path.join(opt_P['path']['val_images'], img_name)
                    img_dir = opt_P['path']['val_images']+"/"+str(current_step)
                    # img_dir = os.path.join(opt_F['path']['val_images'], str(current_step), '_', str(step))
                    util.mkdir(img_dir)
                    # save_lr_path = os.path.join(img_dir, '{:s}_LR.png'.format(img_name))
                    save_img_path = os.path.join(img_dir, '{:s}_LR.png'.format(img_name))
                    util.save_img(lr_img, save_img_path)

                    #sigma = torch.tensor([0]*B).float().view([B, 1, 1, 1]).to(device)
                    sigma = torch.tensor([noise_level]*B).float().view([B, 1, 1, 1]).to(device)
                    SR_it = []
                    ker_it = []
                    
                    #sz_hr = val_data['GT'].size()
                    # sz_hr = LR_img.size()
                    # ker_img = np.zeros(sz_hr)
                    # for indB in range(B):
                    #     ker_cur = b_kernels[indB].detach().cpu()
                    #     ker_img[indB,:,:,:] = np.transpose(util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()),w_h=(sz_hr[3],sz_hr[2])), (2,0,1))
                    # ker_tensor = torch.tensor(ker_img).float().to(device)

                    with torch.no_grad():
                        model_GAN.feed_data(LR_img, val_data['GT'], b_kernels, opt_P['scale'], sigma)
                        model_GAN.test()

                        #SR_img, ker_est, b_check = model_SR(LR_img, b_kernels, opt_P['scale'], sigma, val_data['GT'], SR_it, ker_it) 
                        # ker_est_code = model_SR.netC(val_data['GT'].to(device), LR_img)  
                        # #ker_est = model_SR.netC(b_kernels.view(B,1,H,W)) 
                        # ker_est = model_SR.dec(ker_est_code)
                        # ker_est = ker_est.view(1,21,21)
                    visuals = model_GAN.current_visuals()
                    sr_img = util_image.tensor2uint(visuals['E'])[:,:,::-1]
                    gt_img = util_image.tensor2uint(visuals['H'])[:,:,::-1]
                        
                    #sr_img = util.tensor2img(ker_est.detach().cpu())  # uint8
                    #gt_img = util.tensor2img(b_kernels.detach().cpu())  # uint8

                    step = 8
                    # for indSR in range(8): 
                    #     sr_img = util.tensor2img(SR_it[indSR])
                    #     save_img_path = os.path.join(img_dir, '{:s}_{:d}_{:d}.png'.format(img_name, current_step, indSR))
                    #     util.save_img(sr_img, save_img_path)
            
                    #     ker_cur = ker_it[indSR]
                    #     ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
                    #     save_ker_path = os.path.join(img_dir, 'ker_{:d}_{:d}.png'.format(current_step, indSR))
                    #     util.save_img(ker_img, save_ker_path)
                    
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, 8))
                    util.save_img(sr_img, save_img_path)
                          
                    ker_cur = b_kernels.detach().cpu()
                    ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
                    save_ker_path = os.path.join(img_dir, '{:s}_ker_{:d}_org.png'.format(img_name,0))
                    util.save_img(ker_img, save_ker_path)
                    
                    ker_est = visuals['K']
                    ker_cur = ker_est.detach().cpu()
                    ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
                    save_ker_path = os.path.join(img_dir, '{:s}_ker_{:d}_est.png'.format(img_name,0))
                    util.save_img(ker_img, save_ker_path)
                    
                    # calculate PSNR
                    #crop_size = opt_P['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    step_psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    step_ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)                    
                    #logger.info('img:{:s}'.format(img_name))
                    #logger.info(
                    #    '<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, psnr: {:.6f}'.format(epoch, current_step, step,
                    #                                                                img_name, step_psnr))
                    img_psnr += step_psnr
                    img_ssim += step_ssim
                    idx += 1
                    #avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    #avg_ssim += util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)                    
                    ker_img     = b_kernels[0,:,:].detach().cpu().numpy()
                    ker_img_est = ker_img
                    #ker_img_est = ker_est[0,0,:,:].detach().cpu().numpy()
                    avg_psnr += util.calculate_psnr(ker_img*255 , ker_img_est*255 )
                    avg_mse  += util.calculate_mse (ker_img*255 , ker_img_est*255 )
                      # # avg_signle_img_psnr = single_img_psnr / 1
                    # # logger.info(
                    # #     '<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, average psnr: {:.6f}'.format(epoch, current_step, step,
                    # #                                                                 img_name, avg_signle_img_psnr))

                avg_psnr = avg_psnr / idx
                avg_mse  = avg_mse / idx
                img_psnr = img_psnr / idx
                img_ssim = img_ssim / idx

                # log
                logger.info('# Validation # PSNR: {:.6f}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}, step:{:3d}> psnr: {:.6f}, mse: {:.6f}, img psnr: {:.6f}, img ssim: {:.6f},'.format(epoch, current_step, step, avg_psnr, avg_mse, img_psnr, img_ssim))
                # tensorboard logger
                if opt_P['use_tb_logger'] and 'debug' not in opt_P['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)


    tb_logger.close()
    

if __name__ == '__main__':
    main()

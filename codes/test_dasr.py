
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys


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
import cv2

import options.options as option
from utils import util
from data import create_dataloader, create_dataset

import matplotlib.pyplot as plt
from scipy.io import loadmat

from motionblur.motionblur import Kernel 

from torch.autograd import Variable
sys.path.append('/home/hfates/Documents/deep_projects/DeepImage/DASR/')
import utility
from argparse import Namespace
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################
model_no = 88
val_noise =  0.0
is_val_noise = False

sc_final = 4

bicub_down= False

sel_ker = [0,1,2,3,4,5,6,7,8,9,10,11]
#sel_ker = [0, 1,4,6,8,9,10,11]
#sel_ker = [0]
##########################

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn': #Return the name of start method used for starting processes
        mp.set_start_method('spawn', force=True) ##'spawn' is the default on Windows
    rank = int(os.environ['RANK']) #system env process ranks
    num_gpus = torch.cuda.device_count() #Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs) #Initializes the default distributed process group


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
    opt_P = option.parse(args.opt_P, is_train=True)
    opt_C = option.parse(args.opt_C, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt_P = option.dict_to_nonedict(opt_P)
    opt_C = option.dict_to_nonedict(opt_C)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #opt_F = opt_F['sftmd']

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
        util.setup_logger('base', opt_P['path']['log'], 'temp_' + opt_P['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('test', opt_P['path']['log'], 'test_' + opt_P['name'], level=logging.INFO,
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
                from torch.utils.tensorboard import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/test/' + opt_P['name'])
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
    # model_SR = BLDSRnet(opt_P,opt_C)  #load pretrained model of USRNET ##change
    enc= util.PCAEncoder(pca_matrix, cuda=True)
    inv_pca = torch.transpose(enc.weight,0,1)
    
    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        # model_SR.pre.resume_training(resume_state)  # handle optimizers and schedulers  ##change
    else:
        current_step = 0
        start_epoch = 0
        
    #####
    #Kernel Generation
    kernels = loadmat('kernels_12.mat')['kernels']
    motion_kernels=torch.load('../data_samples/blur_kernels/motion.m')
    # motion_kernels=torch.load('/home/asfand/Desktop/asfand/medipol/medipol/motion.m')
    for filt_ind in range(8):
        kernels[0, filt_ind] = kernels[0, filt_ind][2:-2,2:-2]
    for filt_ind in range(4):
        kernels[0, filt_ind+8] = np.array(motion_kernels[filt_ind], dtype=np.float64) 
        
    noise_level = val_noise
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

    #### testing 
    if bicub_down:
        prepro_val = util.USRNetTest_bicubic(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                              kernel=opt_P['kernel_size'], noise=is_val_noise, cuda=True, sig=opt_P['sig'],
                                              sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                              rate_cln=0.2, noise_high=noise_level)
    else:
        prepro_val = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                              kernel=opt_P['kernel_size'], noise=is_val_noise, cuda=True, sig=opt_P['sig'],
                                              sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                              rate_cln=0.2, noise_high=noise_level)
    
    crop_size = opt_P['scale']**2
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(1):
        
        #for filt_ind in range(12):
        for filt_ind in sel_ker:
            current_step += 1

            # validation, to produce ker_map_list(fake)
            if (1):
                avg_psnr = 0.0
                avg_mse  = 0.0
                img_psnr = 0.0
                img_ssim = 0.0
                idx = 0
                #model_DSR.eval() ##change
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
                        
                    LR_img, b_kernels, noise_level_rand = prepro_val(val_data['GT'], filt,  kernel=True)
                
                    #single_img_psnr = 0.0
                    lr_img = util.tensor2img(LR_img) #save LR image for reference
                    B, H, W = 1,21,21 #[B, l, l]
                                       
                    # Save images for reference
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    #img_dir = os.path.join(opt_P['path']['val_images'], img_name)
                    img_dir = opt_P['path']['val_images']+"/ft_scale"+str(sc_final)+"/"+str(current_step)
                    # img_dir = os.path.join(opt_F['path']['val_images'], str(current_step), '_', str(step))
                    util.mkdir(img_dir)
                    # save_lr_path = os.path.join(img_dir, '{:s}_LR.png'.format(img_name))
                    save_img_path = os.path.join(img_dir, '{:s}_LR.png'.format(img_name))
                    # util.save_img(lr_img, save_img_path) #ahmed

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
                    
                    #SR_img = model_DSR(LR_img * 255)
                    with torch.no_grad():
                        SR_img = DASR(LR_img * 255)
                        SR_img /= 255.0
                    # with torch.no_grad():
                    #     SR_img, ker_est, b_check, ker_est_code = model_SR(LR_img, b_kernels, opt_P['scale'], sigma, val_data['GT'], SR_it, ker_it) 
                        # ker_est_code = model_SR.netC(val_data['GT'].to(device), LR_img)  
                        # #ker_est = model_SR.netC(b_kernels.view(B,1,H,W)) 
                    ''' pk trained on kernel code
                    ker_est = model_SR.dec(pkout)
                    ker_est = ker_est.view(1,21,21)
                    '''
                    
                    gt_img = util.tensor2img(val_data['GT'].detach().cpu())  # uint8
                    sr_img = util.tensor2img(SR_img.detach().cpu())  # uint8
                    #sr_img = util.tensor2img(ker_est.detach().cpu())  # uint8
                    #gt_img = util.tensor2img(b_kernels.detach().cpu())  # uint8
                    
                    #The following code creates 0.5 pixel shift             
                    srnew = cv2.resize(sr_img, (sr_img.shape[1]*2, sr_img.shape[0]*2), cv2.INTER_CUBIC)
                    sr_img_sh = 0.5*srnew[1:-1:2,1:-1:2,:] + 0.5*srnew[2::2,2::2,:]
                    #Additional 1 pixel shift for scale=4 (make this 0 for scale=2)
                    shift_no = 0
                    
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
                    util.save_img(sr_img, save_img_path)#ahmed
                    
                    ker_cur = b_kernels.detach().cpu() ## revision
                    # ker_cur = b_check.detach().cpu() ## ?? suspected mistake
                    ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
                    save_ker_path = os.path.join(img_dir, '{:s}_ker_{:d}_org.png'.format(img_name,0))
                    # util.save_img(ker_img, save_ker_path) #ahmed
                    
                    #ker_cur = ker_est.detach().cpu().view(1,21,21)
                    #ker_img_est = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
                    #save_ker_path = os.path.join(img_dir, '{:s}_ker_{:d}_est.png'.format(img_name,0))
                    # util.save_img(ker_img_est, save_ker_path) #ahmed
                    
                    # calculate PSNR
                    #crop_size = opt_P['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    sr_img_sh = sr_img_sh / 255.
                    if not bicub_down:
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    else:
                        cropped_sr_img = sr_img_sh[crop_size+shift_no:-crop_size+shift_no+1, crop_size+shift_no:-crop_size+shift_no+1, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    
                    step_psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    step_ssim  = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                    #logger.info('img:{:s}'.format(img_name))
                    #logger.info(
                    #    '<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, psnr: {:.6f}'.format(epoch, current_step, step,
                    #                                                                img_name, step_psnr))
                    img_psnr += step_psnr
                    img_ssim += step_ssim
                    idx += 1
                    #avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    #avg_ssim += util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                    #ker_img     = b_kernels[0,:,:].detach().cpu().numpy()
                    #ker_img_est = ker_est[0,0,:,:].detach().cpu().numpy()#AHMED
                    #b_check = b_check[0,0,:,:].detach().cpu().numpy()#AHMED
                    #avg_psnr += util.calculate_psnr(ker_img*255 , ker_img_est*255 )
                    #avg_mse += util.calculate_mse (ker_img*255 , ker_img_est*255 )
                    # fig = plt.Figure(figsize=(10,5))
                    # ax = plt.subplot(2,2,1)
                    # plt.imshow(ker_img)
                    # ax.set_title('original kernel')
                    # ax1 = plt.subplot(2,2,2)
                    # #plt.imshow(ker_img_est)
                    # ax1.set_title('est. ker')
                    # ax3 = plt.subplot(2,2,3)
                    # #plt.imshow(b_check)
                    # ax3.set_title('initial est ker')
                    # ax2 = plt.subplot(2,2,4)
                    # plt.imshow(LR_img[0].detach().cpu().numpy().transpose(1,2,0))
                    # plt.show()
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
                logger_val = logging.getLogger('test')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}, step:{:3d}> psnr: {:.6f}, mse: {:.6f}, img psnr: {:.6f}, img ssim: {:.6f},'.format(epoch, filt_ind+1, step, avg_psnr, avg_mse, img_psnr, img_ssim))
                # tensorboard logger
                if opt_P['use_tb_logger'] and 'debug' not in opt_P['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)


    tb_logger.close()
    

if __name__ == '__main__':
    main()


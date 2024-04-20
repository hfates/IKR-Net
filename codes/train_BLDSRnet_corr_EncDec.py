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
from data import create_dataloader, create_dataset
from models import create_model
from models.BLDSRnet_v2 import BLDSRNet_v2 as net
from torch.optim import Adam

import matplotlib.pyplot as plt
from scipy.io import loadmat

from motionblur.motionblur import Kernel 

from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def BLDSRnet(opt_P, opt_C):
    
    model = net(opt_P,opt_C,n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                       nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    #model.load_state_dict(torch.load('../checkpoints/F_13500.pth'), strict=False)
    #model.load_state_dict(torch.load('../checkpoints/F_1000.pth'), strict=False)
    #model.load_state_dict(torch.load('../checkpoints/usrnet.pth'), strict=False)
    #model.eval()
    for key, v in model.named_parameters():
        #print(key)
        if key[0:4]=='netC':
            v.requires_grad = True
        else:
            v.requires_grad = False
    model = model.to(device) 
    return model

def define_optimizer(model):
    G_optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            #print(k)
            G_optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    G_optimizer = Adam(G_optim_params, lr=1e-4, weight_decay=0)
    #G_optimizer = torch.optim.SGD(G_optim_params, lr=1e-3, momentum=0.9)
    return G_optimizer

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_P', type=str, help='Path to option YMAL file of Predictor.')
    parser.add_argument('-opt_C', type=str, help='Path to option YMAL file of Corrector.')
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
        logger.info(option.dict2str(opt_P))
        logger.info(option.dict2str(opt_C))
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
    F_opt=define_optimizer(model_SR.netC)
    F_loss = torch.nn.MSELoss().to(device)
    enc= util.PCAEncoder(pca_matrix, cuda=True)
    inv_pca = torch.transpose(enc.weight,0,1)
    kernels = loadmat('kernels_12.mat')['kernels']
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
    filt_no =  10000
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
               filt_ind = np.random.randint(8)
               filt = kernels[0, filt_ind].astype(np.float64)
               filt = filt[2:-2,2:-2]
        ker_Arr[ind,:,:] = torch.FloatTensor(filt).to(device)
        
    ba_size = opt_C['datasets']['train']['batch_size']
    prepro = util.USRNetTestMotion(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                              kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                              sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                              rate_cln=0.2, noise_high=0.0)

    #### training 
    ind_it = 0
    idxAll = np.random.permutation(filt_no)
        
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt_P['dist']:
            train_sampler.set_epoch(epoch)
        
        if (ind_it>filt_no):
            idxAll = np.random.permutation(filt_no)
            ind_it = 0
        
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            
            model_SR.netC.train()
            #### update learning rate, schedulers
            # model.update_learning_rate(current_step, warmup_iter=opt_P['train']['warmup_iter'])

            #### preprocessing for LR_img and kernel map

            ind_itM = ind_it % (filt_no+1-ba_size)
            b_kernels = ker_Arr[idxAll[ind_itM:ind_itM+ba_size],:,:]
            ind_it += ba_size
 
            # if (0):
            # #if (np.random.randint(2)):                
            #     prepro = util.USRNetPreprocessing(opt_P['scale'], pca_matrix, random=True, para_input=opt_P['code_length'],
            #                                           kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
            #                                           sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=0.5, scaling=3,
            #                                           rate_cln=0.2, noise_high=0.0)
            #     LR_img, ker_map, b_kernels = prepro(train_data['GT'], kernel=True)
            # else:
            #      #HFA
            #     # if (np.random.randint(2)): # Initialise Kernel
            #     #     kernel = Kernel(size=(21, 21), intensity=np.random.rand())
            #     #     # Display kernel
            #     #     #kernel.displayKernel()
            #     #     # Get kernel as numpy array
            #     #     filt = kernel.kernelMatrix     
            #     # else:
            #     #     filt_ind = np.random.randint(8)
            #     #     filt = kernels[0, filt_ind].astype(np.float64)
            #     #     filt = filt[2:-2,2:-2]
                
            #     # prepro = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
            #     #                                           kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
            #     #                                           sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
            #     #                                           rate_cln=0.2, noise_high=0.0)
            LR_img, b_kernels = prepro(train_data['GT'], b_kernels, kernel=True)
            
            #kernel_code = enc(b_kernels) 
            B, H, W = b_kernels.size() #[B, l, l]
            kernel_code = model_SR.enc(b_kernels.view(B,H*W)) 
            
            noise = torch.mul(torch.FloatTensor(np.random.normal(loc=0, scale=1.0, size=b_kernels.size())), 0.01).to(device)
            #b_kernels_N = torch.clamp(noise + b_kernels, min=0, max=1)
            #b_kernels_N = torch.div(b_kernels_N, torch.sum(b_kernels_N, (1,2)).view(b_kernels_N.size()[0],1,1)) 
            #b_kernels_N = b_kernels
            
            #sz_hr = train_data['GT'].size()
            # sz_hr = LR_img.size()
            # ker_img = np.zeros(sz_hr)
            # for indB in range(B):
            #     ker_cur = b_kernels[indB].detach().cpu()
            #     ker_img[indB,:,:,:] = np.transpose(util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()),w_h=(sz_hr[3],sz_hr[2])), (2,0,1))
            # ker_tensor = torch.tensor(ker_img).float().to(device)
            F_opt.zero_grad()
            
            sigma = torch.tensor([0]*B).float().view([B, 1, 1, 1]).to(device)
            #SR_img, ker_img = model_SR(LR_img, b_kernels, opt_P['scale'], sigma)             
            est_ker_map = model_SR.netC(train_data['GT'].to(device), LR_img)
            #ker_est = model_SR.netC(b_kernels_N.view(B,1,H,W).detach())             
            # #update USRNet weights
            # # torch.autograd.set_detect_anomaly(True)
            #fl=F_loss(SR_img,train_data['GT'].to(device))
            #fl=F_loss(ker_est,b_kernels.view(B,1,H,W).to(device))
            fl=F_loss(est_ker_map,kernel_code.to(device))
            fl.backward()
            F_opt.step()
            model_SR.ker.log_dict['F_loss'] = fl.item()

            #### log of model_P
            if current_step % opt_P['logger']['print_freq'] == 0:
                logs = model_SR.ker.get_current_log()
                message = 'Predictor <epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model_SR.ker.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt_P['use_tb_logger'] and 'debug' not in opt_P['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt_P['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                model_SR.netC.eval()
                for _, val_data in enumerate(val_loader):
                    if (0):
                    #if (np.random.randint(2)):
                        prepro_val = util.USRNetPreprocessing(opt_P['scale'], pca_matrix, random=True, para_input=opt_P['code_length'],
                                                        kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                                        sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=0.5, scaling=3,
                                                        rate_cln=0.2, noise_high=0.0)
                        LR_img, ker_map, b_kernels = prepro_val(val_data['GT'], kernel=True)
                    else:
                        if (np.random.randint(2)): # Initialise Kernel
                            kernel = Kernel(size=(21, 21), intensity=np.random.rand())
                            # Display kernel
                            #kernel.displayKernel()
                            # Get kernel as numpy array
                            filt = kernel.kernelMatrix     
                        else:
                            filt_ind = np.random.randint(8)
                            filt = kernels[0, filt_ind].astype(np.float64)
                            filt = filt[2:-2,2:-2]
                            
                        prepro_val = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                                                  kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                                                  sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                                                  rate_cln=0.2, noise_high=0.0)
                        LR_img, b_kernels = prepro_val(val_data['GT'], filt,  kernel=True)
                    
                    single_img_psnr = 0.0
                    lr_img = util.tensor2img(LR_img) #save LR image for reference
                    B, H, W = 1,21,21 #[B, l, l]
                                       
                    # Save images for reference
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    #img_dir = os.path.join(opt_P['path']['val_images'], img_name)
                    img_dir = opt_P['path']['val_images']
                    # img_dir = os.path.join(opt_F['path']['val_images'], str(current_step), '_', str(step))
                    util.mkdir(img_dir)
                    # save_lr_path = os.path.join(img_dir, '{:s}_LR.png'.format(img_name))
                    # util.save_img(lr_img, save_lr_path)

                    sigma = torch.tensor([0]*B).float().view([B, 1, 1, 1]).to(device)
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
                        #SR_img, _ =model_SR(LR_img, b_kernels, opt_P['scale'], sigma, SR_it, ker_it) 
                        ker_est_code = model_SR.netC(val_data['GT'].to(device), LR_img)  
                        #ker_est = model_SR.netC(b_kernels.view(B,1,H,W)) 
                        ker_est = model_SR.dec(ker_est_code)
                        ker_est = ker_est.view(1,21,21)
                    #sr_img = util.tensor2img(SR_img.detach().cpu())  # uint8
                    #gt_img = util.tensor2img(val_data['GT'].detach().cpu())  # uint8
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
                    
                    ker_cur = b_kernels.detach().cpu()
                    ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
                    save_ker_path = os.path.join(img_dir, '{:s}_ker_{:d}_org.png'.format(img_name,0))
                    util.save_img(ker_img, save_ker_path)
                    
                    ker_cur = ker_est.detach().cpu()
                    ker_img = util.tensor2ker(ker_cur, min_max=(ker_cur.min(), ker_cur.max()))
                    save_ker_path = os.path.join(img_dir, '{:s}_ker_{:d}_est.png'.format(img_name,0))
                    util.save_img(ker_img, save_ker_path)
                    # calculate PSNR
                    # crop_size = opt_P['scale']
                    # gt_img = gt_img / 255.
                    # sr_img = sr_img / 255.
                    # cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    # cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    # step_psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    # logger.info(
                    #     '<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, psnr: {:.6f}'.format(epoch, current_step, step,
                    #                                                                 img_name, step_psnr))
                    # single_img_psnr += step_psnr
                    idx += 1
                    # avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                    # avg_signle_img_psnr = single_img_psnr / 1
                    # logger.info(
                    #     '<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, average psnr: {:.6f}'.format(epoch, current_step, step,
                    #                                                                 img_name, avg_signle_img_psnr))

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.6f}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}, step:{:3d}> psnr: {:.6f}'.format(epoch, current_step, step, avg_psnr))
                # tensorboard logger
                if opt_P['use_tb_logger'] and 'debug' not in opt_P['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

            #### save models and training states
            if current_step % opt_P['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model_SR.pre.save(current_step)
                    torch.save(model_SR.state_dict(), r'../checkpoints/F_'+str(current_step)+'.pth')
                    model_SR.pre.save_training_state(epoch, current_step)
                    model_SR.ker.save(current_step)
                    model_SR.ker.save_training_state(epoch, current_step)


    if rank <= 0:
        logger.info('Saving the final model.')
        model_SR.pre.save('newp')
        model_SR.ker.save('newc')
        logger.info('End of Predictor and Corrector training.')
    tb_logger.close()


if __name__ == '__main__':
    main()


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 12:21:37 2021

@author: asfand
"""
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import argparse
import random
import logging
from models.usrnet_modified import USRNet as net
from models import create_model
from utils import util
import options.options as option
from data import create_dataloader, create_dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.optim import Adam
from scipy.io import loadmat
from scipy import ndimage
    
def USRNET():
    
    model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                       nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    model.load_state_dict(torch.load(r'../checkpoints/F_47000_it1.pth'), strict=True)
    model.train()
    for key, v in model.named_parameters():
        v.requires_grad = True
    model = model.to(device) 
    return model

def define_optimizer(model):
    G_optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            G_optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    G_optimizer = Adam(G_optim_params, lr=1e-4, weight_decay=0)
    return G_optimizer
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_P', type=str, help='Path to option YMAL file of Predictor.')
    parser.add_argument('-opt_C', type=str, help='Path to option YMAL file of Corrector.')
    args = parser.parse_args()
    opt_P = option.parse(args.opt_P, is_train=True)
    opt_C = option.parse(args.opt_C, is_train=True)
   
    # convert to NoneDict, which returns None for missing keys
    opt_P = option.dict_to_nonedict(opt_P)
    opt_C = option.dict_to_nonedict(opt_C)
    
    seed = opt_P['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)
    opt_P['dist'] = False
    # opt_F['dist'] = False
    opt_C['dist'] = False
    rank = -1
     
   
   
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
            if version >= 1.1:  # PyTorch 1.1
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
    for phase, dataset_opt in opt_P['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt_P['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
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
    print('load PCA matrix')
    pca_matrix = torch.load('./pca_matrix.pth',map_location=lambda storage, loc: storage)
    print('PCA matrix shape: {}'.format(pca_matrix.shape))
     #### create model
    model_F = USRNET() #load pretrained model of USRNET
    F_opt=define_optimizer(model_F)
    F_loss = torch.nn.L1Loss().to(device)
    model_P = create_model(opt_P)
    model_C = create_model(opt_C)
    enc= util.PCAEncoder(pca_matrix, cuda=True)
    inv_pca = torch.transpose(enc.weight,0,1)   
    
    kernels = loadmat('kernels_12.mat')['kernels']
    filt = kernels[0, 0].astype(np.float64)
#### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model_P.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    
 #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    # load PCA matrix of enough kernel
    print('load PCA matrix')
    pca_matrix = torch.load('pca_matrix.pth',map_location=lambda storage, loc: storage)
    for epoch in range(start_epoch, total_epochs + 1):
        if opt_P['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            model_F.train()
            #### preprocessing for LR_img and kernel map
            # prepro = util.USRNetPreprocessing(opt_P['scale'], pca_matrix, random=True, para_input=opt_P['code_length'],
            #                                       kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
            #                                       sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
            #                                       rate_cln=0.2, noise_high=0.0)
            # LR_img, ker_map, b_kernels = prepro(train_data['GT'], kernel=True)
            
            #HFA
            prepro = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                                      kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                                      sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                                      rate_cln=0.2, noise_high=0.0)
            LR_img, b_kernels = prepro(train_data['GT'], filt, kernel=True)

            #kernel_code = enc(b_kernels) 
            #B, H, W = b_kernels.size() #[B, l, l]
            #B, H, W =(1,21,21)
            B = b_kernels.size()[0] 
            H, W =(21,21)
            
            #torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))
            # b_check = torch.bmm( ker_map.expand((B, )+ker_map.size()), inv_pca.expand((B, ) + inv_pca.size())).view((B, -1))
            # b_check = torch.add(b_check[0], 1/(H*W)).view((B, H , W))
            
            # ff = np.array(b_kernels.cpu())
            # plt.imshow(ff.T)
            # plt.show()   
            
            # ff = np.array(b_check.cpu())
            # plt.imshow(ff.T)
            # plt.show()
            #model_P.feed_data(LR_img, ker_map)
            model_P.feed_data(LR_img)
            # model_P.optimize_parameters(current_step)
            model_P.test()
            P_visuals = model_P.get_current_visuals()
            est_ker_map = P_visuals['Batch_est_ker_map']
            #est_ker_map = b_kernels
            
            SR_img= torch.nn.functional.interpolate(LR_img, scale_factor=opt_P['scale'], mode='nearest')
           
            #### log of model_P
            if current_step % opt_P['logger']['print_freq'] == 0:
                logs = model_P.get_current_log()
                message = 'Predictor <epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model_P.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt_P['use_tb_logger'] and 'debug' not in opt_P['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)


            #### training Corrector   7 iterations
            usr_itr=8
            for step in range(opt_C['step']):
                # print('step',step)
                # b_check = torch.bmm( est_ker_map.to(device).expand((B, )+est_ker_map.size()), inv_pca.expand((B, ) + inv_pca.size())).view((B, -1))
                # b_check = torch.add(b_check[0], 1/(H*W)).view((B, H , W))  
                b_check = torch.bmm( est_ker_map.to(device).expand((B, )+est_ker_map.size()), inv_pca.expand((B, ) + inv_pca.size())).view((B, -1))
                b_check = torch.add(b_check[0], 1/(H*W)).view((B, H , W))    
                 #test USRNET for corresponding SR image
                b_check = b_check.view(B,1,H,W)
           
                # est_ker_map=util.single2tensor4(est_ker_map[..., np.newaxis]).to(device)
                sigma = torch.tensor([0]*B).float().view([B, 1, 1, 1]).to(device)
                 
                _,FB,FBC,F2B,FBFy,ab=model_F(LR_img, b_check,opt_P['scale'],sigma)
                SR_img=model_F.d(SR_img.detach(), FB, FBC, F2B, FBFy, ab[:, step:step+1, ...], opt_P['scale'])
                SR_img = model_F.p(torch.cat((SR_img, ab[:, step+usr_itr:step+usr_itr+1, ...].repeat(1, 1, SR_img.size(2), SR_img.size(3))), dim=1))
             
                # #update USRNet weights
                F_opt.zero_grad()
                # # torch.autograd.set_detect_anomaly(True)
                fl=F_loss(SR_img,train_data['GT'].to(device))
                fl.backward()
                F_opt.step()
                
                # train corrector given SR image and estimated kernel map
                #model_C.feed_data(SR_img, est_ker_map ,train_data['GT'],ker_map,F_opt)
                model_C.feed_data(SR_img, est_ker_map ,train_data['GT'], None,F_opt)
                # model_C.optimize_parameters(current_step)
                model_C.test(fl)
                C_visuals = model_C.get_current_visuals()
                est_ker_map = C_visuals['Batch_est_ker_map']
                #est_ker_map = b_kernels
                
                #### log of model_C
                if current_step % opt_C['logger']['print_freq'] == 0:
                    logs = model_C.get_current_log()
                    message = 'Corrector <epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                        epoch, current_step, model_C.get_current_learning_rate())
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        # tensorboard logger
                        if opt_C['use_tb_logger'] and 'debug' not in opt_C['name']:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        #HFA
                        if step==0:
                            logger.info(message)
                            
            #update USRNet weights
            # F_opt.zero_grad()
            # # # torch.autograd.set_detect_anomaly(True)
            # fl=F_loss(SR_img,train_data['GT'].to(device))
            # fl.backward()
            # F_opt.step()
            
            # validation, to produce ker_map_list(fake)
            if current_step % opt_P['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                model_F.eval()
                idx = 0
                for _, val_data in enumerate(val_loader):
                    # downsampling the image
                    # prepro = util.USRNetPreprocessing(opt_P['scale'], pca_matrix, random=True, para_input=opt_P['code_length'],
                    #                                 kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                    #                                 sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                    #                                 rate_cln=0.2, noise_high=0.0)
                    # LR_img, ker_map = prepro(val_data['GT'])
                    
                    #HFA
                    prepro = util.USRNetTest(opt_P['scale'], pca_matrix, random=True  , para_input=opt_P['code_length'],
                                                              kernel=opt_P['kernel_size'], noise=False, cuda=True, sig=opt_P['sig'],
                                                              sig_min=opt_P['sig_min'], sig_max=opt_P['sig_max'], rate_iso=1.0, scaling=3,
                                                              rate_cln=0.2, noise_high=0.0)
                    LR_img, b_kernels = prepro(val_data['GT'], filt,  kernel=True)
                    
                    single_img_psnr = 0.0
                    lr_img = util.tensor2img(LR_img) #save LR image for reference
                    B, H, W = 1,21,21 #[B, l, l]
                    # valid Predictor
                    model_P.feed_data(LR_img)
                    model_P.test()
                    P_visuals = model_P.get_current_visuals()
                    est_ker_map = P_visuals['Batch_est_ker_map']
                    #est_ker_map = b_kernels
                    SR_img= torch.nn.functional.interpolate(LR_img, scale_factor=opt_P['scale'], mode='nearest')

                    # Save images for reference
                    img_name = os.path.splitext(os.path.basename(val_data['GT_path'][0]))[0]
                    img_dir = os.path.join(opt_P['path']['val_images'], img_name)
                    # img_dir = os.path.join(opt_F['path']['val_images'], str(current_step), '_', str(step))
                    util.mkdir(img_dir)
                    save_lr_path = os.path.join(img_dir, '{:s}_LR.png'.format(img_name))
                    util.save_img(lr_img, save_lr_path)
                   
                    # 7 iterations
                    
                    for step in range(opt_C['step']):
                        
                        #idx += 1
                        b_check = torch.bmm( est_ker_map.to(device).expand((B, )+est_ker_map.size()), inv_pca.expand((B, ) + inv_pca.size())).view((B, -1))
                        b_check = torch.add(b_check[0], 1/(H*W)).view((B, H , W))   
                        b_check = b_check.view(B,1,H,W)
                        sigma = torch.tensor(0).float().view([1, 1, 1, 1]).to(device)
                        with torch.no_grad():  
                            _,FB,FBC,F2B,FBFy,ab=model_F(LR_img, b_check,opt_P['scale'],sigma)
                            SR_img=model_F.d(SR_img, FB, FBC, F2B, FBFy, ab[:, step:step+1, ...], opt_P['scale'])
                            SR_img = model_F.p(torch.cat((SR_img, ab[:, step+usr_itr:step+usr_itr+1, ...].repeat(1, 1, SR_img.size(2), SR_img.size(3))), dim=1))
                        step += 1
                        # est_ker_map=util.single2tensor4(est_ker_map[..., np.newaxis]).to(device)
                        # Test USRNET to produce SR images
                        # SR_img=model_F(LR_img, b_check,opt_P['scale'],sigma)
                       
                        model_C.feed_data(SR_img,est_ker_map)
                        model_C.test()
                        C_visuals = model_C.get_current_visuals()
                        est_ker_map = C_visuals['Batch_est_ker_map']
                        #est_ker_map = b_kernels
                        
                        sr_img = util.tensor2img(SR_img.detach().cpu())  # uint8
                        gt_img = util.tensor2img(val_data['GT'].detach().cpu())  # uint8

                        save_img_path = os.path.join(img_dir, '{:s}_{:d}_{:d}.png'.format(img_name, current_step, step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        crop_size = opt_P['scale']
                        gt_img = gt_img / 255.
                        sr_img = sr_img / 255.
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                        step_psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                        # logger.info(
                        #     '<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, psnr: {:.6f}'.format(epoch, current_step, step,
                        #                                                                 img_name, step_psnr))
                        single_img_psnr += step_psnr
                        #HFA:
                        if step==opt_C['step']:
                            idx += 1
                            avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                    avg_signle_img_psnr = single_img_psnr / step
                    logger.info(
                        '<epoch:{:3d}, iter:{:8,d}, step:{:3d}> img:{:s}, average psnr: {:.6f}'.format(epoch, current_step, step,
                                                                                    img_name, avg_signle_img_psnr))

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
                    model_P.save(current_step)
                    torch.save(model_F.state_dict(), r'../checkpoints/F_'+str(current_step)+'.pth')
                    model_P.save_training_state(epoch, current_step)
                    model_C.save(current_step)
                    model_C.save_training_state(epoch, current_step)

    
    if rank <= 0:
        logger.info('Saving the final model.')
        model_P.save('newp')
        model_C.save('newc')
        logger.info('End of Predictor and Corrector training.')
    tb_logger.close()

    
if __name__ == '__main__':
    main()


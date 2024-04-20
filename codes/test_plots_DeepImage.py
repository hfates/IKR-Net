#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:39:24 2021

@author: hfates
"""
import matplotlib.pyplot as plt
indK = 0

b_in = k.view(B,H,W)
b_est = Xest_S.view(B,H,W)

ffT = b_in.clone()
ff = b_est.clone()
for indK in range(1):
    ffT = np.array(ffT[indK].detach().cpu())
    ff = np.array(ff[indK].detach().cpu())
    plt.imshow(ffT)
    plt.show()
    plt.imshow(ff)
    plt.show()
    # print(np.max(np.abs(ff[0]-ffT)))
    # print(np.max(np.abs(ffT)))
    # print(np.max(np.abs(ff[0])))
    # print(np.sum(ff[0]))
    
    
opt_PA = opt_P.copy() 
opt_PA['path']['pretrain_model_G'] = '../checkpoints/latest_Pker_68000.pth'
dd = create_model(opt_PA)
dd.feed_data(SR_img)
bker = dd.netG(dd.var_L)

############################################################
LR_interp = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

b_check = HR_img.to(device).detach()
#b_check = k.view(B,1,H,W)
#FB = p2o(b_check.detach(), (w*sf, h*sf))
otf = torch.zeros(b_check.shape).type_as(b_check)
otf.copy_(b_check)
otf = torch.rfft(otf, 2, onesided=False)
n_ops = torch.sum(torch.tensor(b_check.shape).type_as(b_check) * torch.log2(torch.tensor(b_check.shape).type_as(b_check)))
otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(b_check)
FB = otf

FBC = cconj(FB, inplace=False)
F2B = r2c(cabs2(FB))
STy = upsample(x, sf=sf)
FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
 
aa = []
Xest_S = torch.zeros((k.shape[1],k.shape[2])).type_as(k)
for i in range(50):
#kv = self.dec(est_ker_map).view(B,1,H,W)
    kv = Xest_S.view(B,1,H,W) 
    #kv = k.view(B,1,H,W) 
    otf = torch.zeros(SR_img.shape[:-2] + SR_img.shape[2:]).type_as(kv)
    otf[...,:kv.shape[2],:kv.shape[3]].copy_(kv)
    for axis, axis_size in enumerate(kv.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    
    #ker_img = self.d(k, FB, FBC, F2B, FBFy, ab[:, step:step+1, ...], self.opt_P['scale'])
    step = 0
    alpha = 10*ab[:, step:step+1, ...]
    
    FR = FBFy + torch.rfft(alpha*otf, 2, onesided=False)
    x1 = cmul(FB, FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = cdiv(FBR, csum(invW, alpha))
    FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
    FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
    Xest = torch.irfft(FX, 2, onesided=False)
    
    for axis, axis_size in enumerate(kv.shape[2:]):
        Xest = torch.roll(Xest, +int(axis_size / 2), dims=axis+2)
    #plt.imshow((Xest[0,0,0:21,0:21].detach().cpu().numpy()))
    #plt.imshow((Xest[0,1,:,:].detach().cpu().numpy()))
    Xest_SAVE = Xest[0,0,0:21,0:21] 
    XS = Xest_SAVE.sum()
    Xest_SAVE = Xest_SAVE - XS/(441) + 1/(441)
    #plt.imshow((Xest_SAVE.detach().cpu().numpy()))
    aa.append(Xest_SAVE.max())
    
    Xest_S = Xest_SAVE.clone()


#####################
import scipy.ndimage as ndimage
hr_norm= (HR_img.cpu().numpy()*255).astype(np.uint8)
hr_blured = ndimage.filters.convolve((hr_norm[0]).transpose(1,2,0), k.view(H,W).cpu().numpy()[...,np.newaxis], mode='wrap').transpose(2,0,1)/255
hr_blured_var = torch.from_numpy(np.ascontiguousarray(hr_blured))  

FBFy = cdiv(torch.rfft(hr_blured_var.to(device).view(1,hr_blured_var.size()[0],hr_blured_var.size()[1],hr_blured_var.size()[2]), 2, onesided=False), FB)
#FBFy = cdiv(torch.rfft(LR_interp, 2, onesided=False), (FB+100))
Xest = torch.irfft(FBFy, 2, onesided=False)
for axis, axis_size in enumerate(kv.shape[2:]):
    Xest = torch.roll(Xest, +int(axis_size / 2), dims=axis+2)
plt.imshow((Xest[0,1,0:21,0:21].view(21,21).detach().cpu().numpy()))

##############################
ss=HR_img.size()
plt.imshow(np.transpose(HR_img.view(3,ss[2],ss[3]).detach().cpu().numpy(),(1,2,0)))
plt.imshow(np.transpose(STy.view(3,ss[2],ss[3]).detach().cpu().numpy(),(1,2,0)))

plt.imshow(np.transpose(LR_interp.view(3,ss[2],ss[3]).detach().cpu().numpy(),(1,2,0)))

plt.imshow((kv.view(21,21).detach().cpu().numpy()))
plt.imshow((Xest_S.detach().cpu().numpy()))

plt.imshow(np.transpose(otf.view(3,ss[2],ss[3]).detach().cpu().numpy(),(1,2,0)))
plt.imshow((otf[0,1,:,:].view(ss[2],ss[3]).detach().cpu().numpy()))



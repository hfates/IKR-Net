import torch
import torch.nn as nn
import models.usrnet_basicblock as B
import numpy as np

from models import create_model
from models.network_swinir import SwinIR as netSwin
from utils import util
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt
 
print('load PCA matrix')
pca_matrix = torch.load('pca_matrix.pth',map_location=lambda storage, loc: storage)
enc= util.PCAEncoder(pca_matrix, cuda=True)
inv_pca = torch.transpose(enc.weight,0,1)  
    
"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = netSwin(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = netSwin(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = netSwin(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = netSwin(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = netSwin(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = netSwin(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'jpeg_car':
        model = netSwin(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 007 color image denoising - light
    elif args.task == 'color_dn_light':
        model = netSwin(upscale=1, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 007 color image denoising - light
    elif args.task == 'color_dn_lighter':
        model = netSwin(upscale=1, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[4, 4, 4, 4],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'


    pretrained_model = torch.load(args.model_path)
    #model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=False)

    return model

def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a/y, b/y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft(t, 2)


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    return torch.ifft(t, 2)


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        if upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x



class ResUKerNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUKerNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=True, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=True, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=True, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=True, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=True, mode='C'+act_mode+'C') for _ in range(nb)])

        if upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=True, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=True, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=True, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=True, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=True, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=True, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=True, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        FR = FBFy + torch.rfft(alpha*x, 2, onesided=False)
        x1 = cmul(FB, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, alpha))
        FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
        Xest = torch.irfft(FX, 2, onesided=False)

        return Xest

"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataKerNet(nn.Module):
    def __init__(self):
        super(DataKerNet, self).__init__()

    def forward(self, kv, x, FB, FBC, F2B, FBFy, alpha, sf, SR_img):

        #B = x.size()[0] 
        H, W =(21,21)
        
        #kv = Xest_S.view(B,1,H,W) 
        #kv = k.view(B,1,H,W) 
        otf = torch.zeros(SR_img.shape[:-2] + SR_img.shape[2:]).type_as(kv)
        otf[...,:kv.shape[2],:kv.shape[3]].copy_(kv)
        for axis, axis_size in enumerate(kv.shape[2:]):
            otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
        
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
        Xest_SAVE = Xest[:,0:1,0:H,0:W] 
        XS = torch.sum(Xest_SAVE, axis=(1,2,3))/(H*W) - 1/(H*W)
        Xest_SAVE = Xest_SAVE - XS.view((XS.size()[0],1,1,1))
        
        return Xest_SAVE

"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x

class encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden1 = nn.Linear(in_features=kwargs["input_shape"], out_features=400)
        self.encoder_hidden2 = nn.Linear(in_features=400, out_features=225)
        self.encoder_hidden3 = nn.Linear(in_features=225, out_features=100)
        self.encoder_hidden4 = nn.Linear(in_features=100, out_features=25)
        self.encoder_output = nn.Linear(in_features=25, out_features=10)

    def forward(self, features):
        out = torch.relu(self.encoder_hidden1(features))
        out = torch.relu(self.encoder_hidden2(out))
        out = torch.relu(self.encoder_hidden3(out))
        out = torch.relu(self.encoder_hidden4(out))
        encoded = torch.sigmoid(self.encoder_output(out))
        return encoded

class decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder_hidden1 = nn.Linear(in_features=10, out_features=25)
        self.decoder_hidden2 = nn.Linear(in_features=25, out_features=100)
        self.decoder_hidden3 = nn.Linear(in_features=100, out_features=225)
        self.decoder_hidden4 = nn.Linear(in_features=225, out_features=400)
        self.decoder_output = nn.Linear(in_features=400, out_features=441)

    def forward(self, encoded):
        out =  torch.relu(self.decoder_hidden1(encoded))
        out =  torch.relu(self.decoder_hidden2(out))
        out =  torch.relu( self.decoder_hidden3(out))
        out =  torch.relu( self.decoder_hidden4(out))
        reconstructed = torch.sigmoid( self.decoder_output(out))
        return reconstructed

class enc_dec(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.enc = encoder(input_shape=441).to(device)
        self.dec = decoder().to(device)
        # self.enc.load_state_dict(torch.load('../checkpoints/model2/9900_encoder_pth'), strict=True)   
        # self.dec.load_state_dict(torch.load('../checkpoints/model2/9900_decoder_pth'), strict=True)    

    def forward(self, inp_ker):
        ker_code =  self.enc(inp_ker.view((inp_ker.size()[0],441))) 
        out_ker  =  self.dec(ker_code)

        return out_ker.view((out_ker.size()[0],1,21,21))
    
"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""


class BLDSRNet_v2(nn.Module):
    def __init__(self, args, opt_P, opt_C, n_iter=1, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='BR', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(BLDSRNet_v2, self).__init__()

        self.pre = create_model(opt_P)
        self.netP= self.pre.netG
        self.enc= encoder(input_shape=441).to(device)
        self.enc.load_state_dict(torch.load('../checkpoints/model2/9900_encoder_pth'), strict=True)   

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=8*2, channel=h_nc)
        self.ker = create_model(opt_C)
        self.netC = self.ker.netG
        #self.netC = enc_dec().to(device)
        #self.netC = Estimator()
        
        self.dec = decoder().to(device)
        self.dec.load_state_dict(torch.load('../checkpoints/model2/9900_decoder_pth'), strict=True)   
        
        self.n = n_iter
        self.opt_P = opt_P
        self.opt_C = opt_C

    def forward(self, x, k, sf, sigma, HR_img=[], SR_it=[], ker_it=[]):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''
        #self.pre.feed_data(x)
        #b_check = self.netP(self.pre.var_L)
        est_ker_map = self.netP(x)
        
         # initialization & pre-calculation
        w, h = x.shape[-2:]
        SR_img = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
        #SR_img= torch.nn.functional.interpolate(LR_img, scale_factor=opt_P['scale'], mode='nearest')
       
        # hyper-parameter, alpha & beta
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))       
        B = x.size()[0] 
        H, W =(21,21)
        
        #est_ker_map = enc(b_check.view(B,H,W))
        b_check = self.dec(est_ker_map).view(B,1,H,W)
        #b_check = k.view(B,1,H,W)
        # indK = 0
        # ffT = k.clone()
        # ffT = np.array(ffT[indK].cpu())
        # plt.imshow(ffT)
        # plt.show()
        #ker_it.append(k)
        
        for step in range(8): #range(self.opt_C['step']):
            # print('step',step)   
             #test USRNET for corresponding SR image
                        
            FB = p2o(b_check.detach(), (w*sf, h*sf))
            #FB = p2o(b_check, (w*sf, h*sf))
            FBC = cconj(FB, inplace=False)
            F2B = r2c(cabs2(FB))
            STy = upsample(x, sf=sf)
            FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
             
            SR_img = self.d(SR_img, FB, FBC, F2B, FBFy, ab[:, step:step+1, ...], self.opt_P['scale'])
            SR_img = self.p(torch.cat((SR_img, ab[:, step+self.n:step+self.n+1, ...].repeat(1, 1, SR_img.size(2), SR_img.size(3))), dim=1))
            
            # SR_it.append(SR_img.clone().detach().cpu())
            # ker_it.append(b_check.clone().detach().cpu())
            # initialization & pre-calculation
           
            # #update USRNet weights
            # F_opt.zero_grad()
            # # # torch.autograd.set_detect_anomaly(True)
            #fl=F_loss(SR_img,train_data['GT'].to(device))
            # fl.backward()
            # F_opt.step()
            
            # # train corrector given SR image and estimated kernel map
            # #self.ker.feed_data(SR_img, est_ker_map ,train_data['GT'], None,F_opt)
            # est_ker_map = self.netC(SR_img, est_ker_map)
            
            # #self.ker.feed_data(SR_img, est_ker_map)
            # # model_C.optimize_parameters(current_step)
            # #self.ker.test()
            # #C_visuals = self.ker.get_current_visuals()
            # #est_ker_map = C_visuals['Batch_est_ker_map']
            
            # # indK = 0
            # # #ffT = np.array(b_kernels[indK].cpu())
            # # ff = b_check[indK].clone()
            # # ff = np.array(ff.detach().cpu())
            # # plt.imshow(ff[0])
            # # plt.show()
    
            # b_check = torch.bmm(est_ker_map.expand((B, )+est_ker_map.size()), inv_pca.expand((B, ) + inv_pca.size())).view((B, -1))
            # #b_check = torch.bmm( est_ker_map.to(device), inv_pca.expand((B, ) + inv_pca.size())).view((B, -1))
            # b_check = torch.add(b_check[0], 1/(H*W)).view((B, H , W))    
            # b_check = b_check.view(B,1,H,W)
            
            if step >= 6:
                est_ker_map = self.netC(SR_img.detach(), x)
                b_check = self.dec(est_ker_map).view(B,1,H,W)
            
            #b_check = k.view(B,1,H,W)
            
        return SR_img, b_check, est_ker_map

class SwinKerNet(nn.Module):
    def __init__(self, args, opt_P, opt_C, n_iter=1, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='BR', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(SwinKerNet, self).__init__()

        self.pre = create_model(opt_P)
        self.netP= self.pre.netG
        self.enc= encoder(input_shape=441).to(device)
        self.enc.load_state_dict(torch.load('../checkpoints/model_ker_code_Asfand/9900_encoder_pth'), strict=True)   

        self.d = DataNet()
        #self.p = ResUNet(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=8*2, channel=h_nc)

        #self.dker = DataKerNet()
        #self.pk = ResUKerNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.ker = create_model(opt_C)
        self.netC = self.ker.netG
        #self.netC = enc_dec().to(device)
        #self.netC = Estimator()
        
        self.dec = decoder().to(device)
        self.dec.load_state_dict(torch.load('../checkpoints/model_ker_code_Asfand/9900_decoder_pth'), strict=True)   
        
        self.swin = define_model(args)
        self.swin = self.swin.to(device)
        
        self.n = n_iter
        self.opt_P = opt_P
        self.opt_C = opt_C

    def forward(self, x, k, sf, sigma, HR_img=[], SR_it=[], ker_it=[]):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''
        #self.load_state_dict(torch.load('../experiments/full_usr_BSD68_ALL/save_models/USRNet_fine_tuned/F_160000.pth'), strict=False)
        #self.load_state_dict(torch.load('../experiments/results_bld_ker/models/F_42000.pth'), strict=True)
        
        # self.load_state_dict(torch.load('../experiments/train_bldsr_init_1it/models/F_22000.pth'), strict=False)
        # pretrained_dict = torch.load('../experiments/results_bld_ker/models/F_42000.pth')
        # pretrained_dict = {k[:]: v for (k, v) in pretrained_dict.items() if k[:2]=='pk'}
        # self.load_state_dict(pretrained_dict, strict=False)
        
        #self.pre.feed_data(x)
        #b_check = self.netP(self.pre.var_L)
        #est_ker_map = self.netP(x)
        
         # initialization & pre-calculation
        w, h = x.shape[-2:]
        STy = upsample(x, sf=sf)
        SR_img = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
        #SR_img= torch.nn.functional.interpolate(LR_img, scale_factor=opt_P['scale'], mode='nearest')
       
        B = x.size()[0] 
        H, W =(21,21)
        
        #est_ker_map = enc(b_check.view(B,H,W))
        #b_init = self.dec(est_ker_map).view(B,1,H,W)
        b_init = k.view(B,1,H,W)
        b_check = b_init
        # indK = 0
        # ffT = k.clone()
        # ffT = np.array(ffT[indK].cpu())
        # plt.imshow(ffT)
        # plt.show()
        #ker_it.append(k)
        
        # b_ker = HR_img.to(device).detach()
        # otf = torch.zeros(b_ker.shape).type_as(b_ker)
        # otf.copy_(b_ker)
        # otf = torch.rfft(otf, 2, onesided=False)
        # n_ops = torch.sum(torch.tensor(b_ker.shape).type_as(b_ker) * torch.log2(torch.tensor(b_ker.shape).type_as(b_ker)))
        # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(b_ker)
        # FBk = otf
        # FBCk = cconj(FBk, inplace=False)
        # F2Bk = r2c(cabs2(FBk))
        # FBFyk = cmul(FBCk, torch.rfft(STy, 2, onesided=False))
         
        #aa = []
        Xest_S = torch.zeros((k.shape[0], 1, k.shape[1],k.shape[2])).type_as(k)
        #Xest_S = b_check
        
        # hyper-parameter, alpha & beta
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))       
        abker = 10*ab[:, 0:1, ...]
        abker[0,0,...] = 0.4
        for step in range(8): #range(self.opt_C['step']):
            # print('step',step)   
             #test USRNET for corresponding SR image
            FB = p2o(b_check.detach(), (w*sf, h*sf))
            #FB = p2o(b_check, (w*sf, h*sf))
            FBC = cconj(FB, inplace=False)
            F2B = r2c(cabs2(FB))
            #STy = upsample(x, sf=sf)
            FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
             
            if step==4:
                SR_inp = SR_img.detach()
            else:
                SR_inp = SR_img
            SR_img = self.d(SR_inp, FB, FBC, F2B, FBFy, ab[:, step:step+1, ...], self.opt_P['scale'])
            #SR_img = self.p(torch.cat((SR_img, ab[:, step+self.n:step+self.n+1, ...].repeat(1, 1, SR_img.size(2), SR_img.size(3))), dim=1))
            SR_img = self.swin(SR_img)
        #     if step<7:
        #         continue
        #     # if step >= 6:
        #     #     est_ker_map = self.netC(SR_img.detach(), x)
        #     #     b_check = self.dec(est_ker_map).view(B,1,H,W)
                     
        #     #Kernel estimation
        #     b_ker = SR_img.detach()
        #     #b_ker = SR_img
        #     otf = torch.zeros(b_ker.shape).type_as(b_ker)
        #     otf.copy_(b_ker)
        #     otf = torch.rfft(otf, 2, onesided=False)
        #     n_ops = torch.sum(torch.tensor(b_ker.shape).type_as(b_ker) * torch.log2(torch.tensor(b_ker.shape).type_as(b_ker)))
        #     otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(b_ker)
        #     FBk = otf
        #     FBCk = cconj(FBk, inplace=False)
        #     F2Bk = r2c(cabs2(FBk))
        #     FBFyk = cmul(FBCk, torch.rfft(STy, 2, onesided=False))        

        # # abker = 10*ab[:, 0:1, ...]
        # # abker[0,0,...] = 0.4
        # # for step in range(1):
        #     Xest_S = self.dker(Xest_S, x, FBk, FBCk, F2Bk, FBFyk, abker, self.opt_P['scale'], SR_img)
        #     #est_ker_map = self.enc(Xest_S.contiguous().view(B,H*W))
        #     #Xest_S = self.dec(est_ker_map).view(B,1,H,W)
        #     #plt.imshow((Xest_S.view(21,21).detach().cpu().numpy()))
        #     #Xest_S = torch.cat((Xest_S,Xest_S,Xest_S), dim=1)
        #     #if (step==0):
        #     Xest_S = self.pk(Xest_S)
                
        #     XS = torch.sum(Xest_S, axis=(1,2,3))/(H*W) - 1/(H*W)
        #     Xest_S = Xest_S - XS.view((XS.size()[0],1,1,1))
        #     #XS = torch.sum(Xest_S, axis=(1,2,3))
        #     #Xest_S = Xest_S / XS.view((XS.size()[0],1,1,1))
        #     #Xest_S = Xest_S[:,0:1,:,:]    
        #     b_check = Xest_S

        # #Xest_S = k.view(B,1,H,W)
        # self.load_state_dict(torch.load('../checkpoints/usrnet.pth'), strict=False)
        # #self.load_state_dict(torch.load('../experiments/full_usr_BSD68_ALL/save_models/USRNet_fine_tuned/F_160000.pth'), strict=False)
        # SR_img = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
        # # hyper-parameter, alpha & beta
        # ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))       
        # for step in range(8): #range(self.opt_C['step']):
        #     # print('step',step)   
        #       #test USRNET for corresponding SR image
        #     FB = p2o(Xest_S.detach(), (w*sf, h*sf))
        #     #FB = p2o(b_check, (w*sf, h*sf))
        #     FBC = cconj(FB, inplace=False)
        #     F2B = r2c(cabs2(FB))
        #     #STy = upsample(x, sf=sf)
        #     FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
             
        #     SR_img = self.d(SR_img, FB, FBC, F2B, FBFy, ab[:, step:step+1, ...], self.opt_P['scale'])
        #     SR_img = self.p(torch.cat((SR_img, ab[:, step+self.n:step+self.n+1, ...].repeat(1, 1, SR_img.size(2), SR_img.size(3))), dim=1))
 
        #Xest_S = b_check 
        return SR_img, Xest_S, b_init



#################################################################
#################################################################
########## Authors: Charles Laroche & Andrés Almansa ############
#### M2 MVA, Deep Learning for Image Restoration & Synthesis ####
#################################################################

import torch
import torch.fft as fft
import numpy as np

def pad_kernel(k, ksize):
    pad_values = (ksize[0] - k.shape[-2]) // 2, (ksize[1] - k.shape[-1]) // 2
    k = torch.nn.functional.pad(k, (pad_values[1],pad_values[1], pad_values[0],pad_values[0]),
                                     mode='constant', value=0)
    return k

def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NCHW
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    return otf

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


def fft_blur(img, k):
    k = p2o(k, img.shape[-2:])
    img_fft = fft.fft2(img)
    ker_fft = fft.fft2(k)

    res_fft = ker_fft.mul(img_fft)
    res = fft.ifft2(res_fft)
    return res.real

def clamp(tensor, min_val, max_val):
    return torch.max(torch.min(tensor, max_val*torch.ones_like(tensor)), min_val*torch.ones_like(tensor))

def clean_output(x):
    return (clamp(x, -1, 1)+1)/2

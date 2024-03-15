""" Code adapted from AstroDDPM repo. """

import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchcubicspline
import camb
from pixell import enmap, utils


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for i in range(self.n_hidden_layers + 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

class CMBPS(nn.Module):

    def __init__(self, norm_phi=True):
        super().__init__()

        # MLP
        self.mlp = MLP(2, 100, 128, 2)
        CKPT_FOLDER = '/mnt/home/dheurtel/ceph/02_checkpoints/SIGMA_EMULATOR'
        MODEL_ID = 'Emulator_H0_ombh2_1'
        ckpt = torch.load(os.path.join(CKPT_FOLDER, MODEL_ID + '.pt'))
        self.mlp.load_state_dict(ckpt['network'])
        for param in self.mlp.parameters():
            param.requires_grad = False

        # Useful variables
        wn = (256*np.fft.fftfreq(256, d=1.0)).reshape((256,) + (1,) * (2 - 1))
        wn_iso = np.zeros((256,256))
        for i in range(2):
            wn_iso += np.moveaxis(wn, 0, i) ** 2
        wn_iso = np.sqrt(wn_iso)
        indices = np.fft.fftshift(wn_iso).diagonal()[128:] ## The value of the wavenumbers along which we have the power spectrum diagonal
        self.register_buffer("torch_indices", torch.tensor(indices))
        self.register_buffer("torch_wn_iso", torch.tensor(wn_iso, dtype=torch.float32))

        # Normalization of phi
        self.norm_phi = norm_phi
        self.register_buffer("min_phi", torch.tensor([50, 7.5e-3]))
        self.register_buffer("dphi", torch.tensor([40, 49.2e-3]))
    
    def forward(self, phi):
        if self.norm_phi:
            #print(self.dphi, self.min_phi)
            phi = phi*self.dphi + self.min_phi
        phi = (phi - torch.tensor([70, 32e-3]).to(phi.device))/torch.tensor([20,25e-3]).to(phi.device)
        torch_diagonals = self.mlp(phi) ## Shape (batch_size, 128) (128 is the number of wavenumbers along which we have the power spectrum diagonal)
        if phi.ndim == 1:
            torch_diagonals = torch_diagonals.unsqueeze(0)
        torch_diagonals = torch.moveaxis(torch_diagonals, -1, 0) ## Shape (128, batch_size) to be able to use torchcubicspline
        spline = torchcubicspline.NaturalCubicSpline(torchcubicspline.natural_cubic_spline_coeffs(self.torch_indices, torch_diagonals))
        return torch.exp(torch.moveaxis(spline.evaluate(self.torch_wn_iso), -1, 0)) / 12661 # 12661 is the mean PS at fiducial cosmology
    
def normalize_phi(phi):
    """ Normalize phi from [50, 90]x[0.0075, 0.0567] to [0, 1]x[0, 1]"""
    min_phi = torch.tensor([50, 7.5e-3]).to(phi.device)
    dphi = torch.tensor([40, 49.2e-3]).to(phi.device)
    return (phi - min_phi) / dphi

def unnormalize_phi(phi):
    """ Unnormalize phi from [0, 1]x[0, 1] to [50, 90]x[0.0075, 0.0567]"""
    min_phi = torch.tensor([50, 7.5e-3]).to(phi.device)
    dphi = torch.tensor([40, 49.2e-3]).to(phi.device)
    return phi * dphi + min_phi

class CMBPS_norm(CMBPS):
    def __init__(self, norm_phi=True):
        super().__init__()
        self.norm_phi = norm_phi

    
    def forward(self, phi):
        if self.norm_phi:
            phi = phi * self.dphi + self.min_phi
        return super().forward(phi)

def patch_shape_and_wcs(Npix, res):
    ndegree_patch = res * Npix / 60
    box = np.array([[-1, 1], [1, -1]]) * utils.degree * ndegree_patch / 2
    shape, wcs = enmap.geometry(pos=box, res=res*utils.arcmin, proj='car')
    return shape, wcs

def sym_mean(ps):
    return 1/4*(ps + np.rot90(ps, 2) + np.rot90(ps, 1) + np.rot90(ps, 3))

def get_camb_ps(phi):
    pars = camb.CAMBparams()

    fixed_cosmo = True
    zero_r = False

    Npix = 256    # Number of pixels
    res = 8       # Size of a pixel, in arcminutes

    shape_patch, wcs_patch = patch_shape_and_wcs(Npix, res)
    shape = [3, shape_patch[0], shape_patch[1]]

    H0 = phi[0]
    ombh2 = phi[1]
    omch2 = np.random.normal(0.12, 0.0012) if not fixed_cosmo else 0.12
    tau = np.random.normal(0.0544, 0.0073) if not fixed_cosmo else 0.0544
    logas = np.random.normal(3.044, 0.014) if not fixed_cosmo else 3.044
    As = np.exp(logas) / 1e10
    ns = np.random.normal(0.9649, 0.0042) if not fixed_cosmo else 0.9649

    # From Planck 2018 I. paper, see Sect. 5.3
    mnu = np.random.uniform(0.06, 0.1) if not fixed_cosmo else 0.8

    # Inspired from the upper limit of Planck 2018 I. paper
    logr = np.random.uniform(-3, -1.494850021680094)
    if not zero_r:
        r = 10**logr
    else:
        r = 0

    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(4000, lens_potential_accuracy=0)
    pars.WantTensors = True

    results = camb.get_results(pars)
    cl_TT, cl_EE, cl_BB, cl_TE = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)['total'].T

    comp = np.zeros((3, 3, cl_TT.shape[-1]))
    comp[0,0] = cl_TT
    comp[1,1] = cl_EE
    comp[0,1] = cl_TE
    comp[1,0] = cl_TE
    comp[2,2] = cl_BB

    ps_maps = enmap.spec2flat(shape, wcs_patch, enmap.massage_spectrum(comp, shape), 1, mode='constant')
    true_ps = ps_maps[0,0]

    true_ps = np.fft.fftshift(true_ps)
    true_ps[1:,1:] = sym_mean((true_ps)[1:,1:]) # symmetrize the PS

    return np.fft.ifftshift(true_ps)

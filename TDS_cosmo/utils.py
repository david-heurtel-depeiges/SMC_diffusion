import torch
import numpy as np


def get_phi_bounds(device=None):
    phi_min = torch.tensor([50, 7.5e-3]).to(device)
    phi_max = torch.tensor([90, 56.7e-3]).to(device)
    return phi_min, phi_max

def normalize_phi(phi, mode='compact'):
    """ Normalize phi from its bounded domain to 
    - [0, 1]x[0, 1] for mode=='compact'
    - [-inf, inf]x[-inf, inf] for mode=='inf' """
    phi_min, phi_max = get_phi_bounds(device=phi.device)
    dphi = phi_max - phi_min
    norm_phi = (phi - phi_min) / dphi
    if mode == 'compact':
        return norm_phi
    elif mode == 'inf':
        return torch.tan((norm_phi - 0.5)*np.pi)
    elif mode is None:
        return phi
    else:
        raise ValueError(f"Unknown normalization mode {mode}")

def unnormalize_phi(phi, mode='compact'):
    """ Unnormalize phi according to the prescribed mode."""
    phi_min, phi_max = get_phi_bounds(device=phi.device)
    dphi = phi_max - phi_min
    if mode == 'compact':
        return phi * dphi + phi_min
    elif mode == 'inf':
        return (torch.arctan(phi)/np.pi + 0.5) * dphi + phi_min
    elif mode is None:
        return phi
    else:
        raise ValueError(f"Unknown normalization mode {mode}")
    
def gen_x(phi, ps_model, device=None):
    """ Generate a CMB map from the parameters phi."""
    ps = ps_model(phi)
    return torch.fft.ifft2(torch.fft.fft2(torch.randn(ps.shape, device=device))*torch.sqrt(ps)).real

def clamp_phi(phi, box_min=0., box_max=1.):
    """
    Clamp the parameters to the box [box_min, box_max].
    """
    return torch.max(torch.min(phi, box_max*torch.ones_like(phi)), box_min*torch.ones_like(phi))

def sample_prior_phi(n, device=None):
    """
    Sample from the prior distribution on phi.
    """
    phi = torch.rand(n, 2).to(device)/7+1/2
    return clamp_phi(phi, box_min=0, box_max=1)

def log_prior_phi(phi):
    """
    Compute the log prior of the parameters.
    """
    #Box
    logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float()) #gives either 0 or -inf
    for i in range(1, phi.shape[-1]):
        logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())
    #Gaussian
    logp += -0.5 * torch.sum((phi-1/2)**2, dim=-1)/(2*1/7**2)
    return logp


# def log_likelihood_eps_phi_sigma(phi, eps, sigma_2_y, sigma_2_regularization, ps_model):
#     """
#     Compute the log likelihood of the Gaussian model (epsilon | phi).
#     """
#     eps_dim = eps.shape[-1]*eps.shape[-2]
#     ps = ps_model(phi)
#     xf = torch.fft.fft2(eps)
#     sigma_2_y = sigma_2_y.reshape(-1,1,1)
#     sigma_2_regularization = sigma_2_regularization.reshape(-1,1,1)

#     term_pi = -(eps_dim/2) * np.log(2*np.pi)
#     term_logdet = -0.5 * torch.sum(torch.log(sigma_2_y*ps+sigma_2_regularization), dim=(-1, -2)) # The determinant is the product of the diagonal elements of the PS
#     term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / (sigma_2_y*ps+sigma_2_regularization), dim=(-1, -2, -3))/eps_dim # We divide by eps_dim because of the normalization of the FFT
#     return term_pi + term_logdet + term_x

def log_likelihood_eps_phi_sigma(phi, eps, sigma_2_y, rescaling, ps_model, sigma_2_rescaling=1):
    """
    Compute the log likelihood of the Gaussian model (epsilon | phi).
    """
    eps_dim = eps.shape[-1]*eps.shape[-2]
    ps = ps_model(phi)
    xf = torch.fft.fft2(eps)
    sigma_2_y = sigma_2_y.reshape(-1,1,1)
    rescaling = rescaling.reshape(-1,1,1)
    sigma_2_rescaling = sigma_2_rescaling.reshape(-1,1,1)

    term_pi = -(eps_dim/2) * np.log(2*np.pi)
    #print(sigma_2_y.shape, ps.shape, rescaling.shape, sigma_2_rescaling.shape)
    term_logdet = -0.5 * torch.sum(torch.log(sigma_2_y*ps*(1-rescaling)+rescaling*sigma_2_rescaling), dim=(-1, -2)) # The determinant is the product of the diagonal elements of the PS
    term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / (sigma_2_y*ps*(1-rescaling)+rescaling*sigma_2_rescaling), dim=(-1, -2, -3))/eps_dim # We divide by eps_dim because of the normalization of the FFT
    return term_pi + term_logdet + term_x


def log_likelihood_eps_phi_sigma_v2(phi, eps, sigma_2_y, rescaling, ps_model, sigma_2_rescaling=1):
    """
    Compute the log likelihood of the Gaussian model (epsilon | phi).
    """
    eps_dim = eps.shape[-1]*eps.shape[-2]
    ps = ps_model(phi)
    xf = torch.fft.fft2(eps)
    sigma_2_y = sigma_2_y.reshape(-1,1,1)
    rescaling = rescaling.reshape(-1,1,1)

    term_pi = -(eps_dim/2) * np.log(2*np.pi)
    term_logdet = -0.5 * torch.sum(torch.log((sigma_2_y+sigma_2_rescaling*rescaling)*ps), dim=(-1, -2)) # The determinant is the product of the diagonal elements of the PS
    term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / ((sigma_2_y+sigma_2_rescaling*rescaling)*ps), dim=(-1, -2, -3))/eps_dim # We divide by eps_dim because of the normalization of the FFT
    return term_pi + term_logdet + term_x
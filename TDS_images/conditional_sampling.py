import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from utils.utils_img import tensor2uint, clean_output
from utils.scheduler import alpha_beta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DPS with DDPM and intrinsic scale
def tds_sampling_batch(model, nsamples, observation, forward_log_likelihood, scheduler, verbose = False, eta=1.0,time_rescaling=False,
                      batch_size=6,threshold_resampling = 0.9, return_auxiliary = False, eta_schedule = None):
    sample_size = model.config.sample_size
    for param in model.parameters():
        param.requires_grad = False

    display = np.linspace(0, len(scheduler.timesteps)-1, 5).astype(int)
    # Init random noise
    x_t = torch.randn((nsamples, 3, sample_size, sample_size)).to(device)

    t = scheduler.timesteps[0]

    ## Precompute some term for loop initialization
    alpha_t, beta_t, alpha_prod, next_alpha_prod = alpha_beta(scheduler, t)
    eta_t =  beta_t / torch.sqrt(alpha_t)
    guidance_all = torch.zeros_like(x_t)
    epsilon_all = torch.zeros_like(x_t)
    log_p_obs_given_x_t_quadratic = torch.zeros(nsamples).to(device)
    log_p_obs_given_x_t_constant = 1
    log_p_obs_given_x_t_log_factor = 0
    if eta_schedule is not None:
        eta = eta_schedule[0]
    for i in range(0, nsamples, batch_size):
        xtemp = x_t[i:min(i+batch_size, nsamples)].clone().detach().requires_grad_()
        epsilon_t = model(xtemp, t).sample
        epsilon_all[i:min(i+batch_size, nsamples)] = epsilon_t.detach().clone()
        predict = scheduler.step(epsilon_t, t, xtemp)
        x0_hat  = clean_output(predict.pred_original_sample)
        # Guidance
        rescaling_factor = time_rescaling * (1-alpha_prod)/eta
        quadratic, constant, log_factor = forward_log_likelihood(x0_hat.to(torch.float32), observation.to(torch.float32), rescaling_factor)
        #ll = quadratic/constant + log_factor
        #guidance = torch.autograd.grad(-ll.sum(), xtemp)[0]
        guidance = torch.autograd.grad(-quadratic.sum(), xtemp)[0]/constant #helps with numerical stability (most of the time)
        guidance_all[i:min(i+batch_size, nsamples)] = guidance.detach().clone()
        log_p_obs_given_x_t_quadratic[i:min(i+batch_size, nsamples)] = quadratic.detach().clone()
        log_p_obs_given_x_t_constant = constant
        log_p_obs_given_x_t_log_factor = log_factor
    log_weights = (log_p_obs_given_x_t_quadratic.clone()/log_p_obs_given_x_t_constant).to(torch.float32) #We can ignore the log_factor term here only (usefull for numerical stability)
    ess_list = []
    all_timesteps = scheduler.timesteps
    nsteps = len(all_timesteps)-1
    for i in tqdm.tqdm(range(nsteps), disable=not verbose):
        with torch.no_grad():
            weights = torch.softmax(log_weights, dim=0)
            ess = 1/torch.sum(weights**2)
            ess_list.append(ess.item())
            if ess < threshold_resampling * nsamples:
                idx = torch.multinomial(weights, nsamples, replacement=True)
                x_t = x_t[idx]
                log_p_obs_given_x_t_quadratic = log_p_obs_given_x_t_quadratic[idx]
                epsilon_all = epsilon_all[idx]
                guidance_all = guidance_all[idx]
                log_weights = torch.zeros(nsamples).to(device).to(torch.float32)
            ## Now diffusion step
            t = all_timesteps[i]
            predict = scheduler.step(epsilon_all, t , x_t)
            x_prev  = predict.prev_sample
            x_prev_mean = predict.mean_prev_sample

            x_prev = x_prev - eta_t * guidance_all

            var_diffusion = scheduler._get_variance(t)
            log_p_non_cond_log_p_cond = - torch.sum(eta_t*guidance_all*(2*(x_prev - x_prev_mean) + eta_t*guidance_all),dim=(1,2,3)).to(torch.float32)/(2*var_diffusion)
            log_p_obs_given_old_quadratic = log_p_obs_given_x_t_quadratic.clone()
            log_p_obs_given_old_constant = log_p_obs_given_x_t_constant
            log_p_obs_given_old_log_factor = log_p_obs_given_x_t_log_factor
   
        x_t = x_prev.clone().detach()

        t = all_timesteps[i+1]
        alpha_t, beta_t, alpha_prod, next_alpha_prod = alpha_beta(scheduler, t)
        eta_t =  beta_t / torch.sqrt(alpha_t)

        ## Compute terms for next step
        if eta_schedule is not None:
            eta = eta_schedule[i+1]
        
        for i_b in range(0, nsamples, batch_size):
            xtemp = x_t[i_b:min(i_b+batch_size, nsamples)].clone().detach().requires_grad_()
            epsilon_t = model(xtemp, t).sample
            epsilon_all[i_b:min(i_b+batch_size, nsamples)] = epsilon_t.detach().clone()
            predict = scheduler.step(epsilon_t, t, xtemp)
            x0_hat  = clean_output(predict.pred_original_sample)
            # Guidance
            rescaling_factor = time_rescaling * (1-alpha_prod)/eta
            quadratic, constant, log_factor = forward_log_likelihood(x0_hat, observation, rescaling_factor)
            #ll = quadratic/constant + log_factor
            #guidance = torch.autograd.grad(-ll.sum(), xtemp)[0]
            guidance = torch.autograd.grad(-quadratic.sum(), xtemp)[0]/constant #helps with numerical stability (most of the time)
            guidance_all[i_b:min(i_b+batch_size, nsamples)] = guidance.detach().clone()
            log_p_obs_given_x_t_quadratic[i_b:min(i_b+batch_size, nsamples)] = quadratic.detach().clone()
            log_p_obs_given_x_t_constant = constant
            log_p_obs_given_x_t_log_factor = log_factor

        log_weights+= log_p_non_cond_log_p_cond
        #log_weights+= (log_p_obs_given_x_t - log_p_obs_given_old).to(torch.float32)
        #print(log_p_obs_given_x_t_quadratic, log_p_obs_given_x_t_constant, log_p_obs_given_x_t_log_factor)
        diff_log_p_obs = 1/log_p_obs_given_x_t_constant * (log_p_obs_given_x_t_quadratic - log_p_obs_given_old_quadratic*(log_p_obs_given_x_t_constant/log_p_obs_given_old_constant)).to(torch.float32)
        #diff_log_p_obs += (log_p_obs_given_x_t_log_factor - log_p_obs_given_old_log_factor) #We can ignore the log_factor term here only (but can be usefull for numerical stability)
        log_weights += diff_log_p_obs
        if log_weights.isnan().any():
            print('Nan detected')
            print(diff_log_p_obs, log_p_non_cond_log_p_cond)
            break
        if log_weights.isinf().all():
            print('Inf detected')
            print(diff_log_p_obs, log_p_non_cond_log_p_cond)
            break
        if i in display:
            # Show progress of the sampling
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(tensor2uint(x_t[0]))
            ax[0].axis('off')
            ax[0].set_title('t = %d' % t)
            ax[1].imshow(tensor2uint(x0_hat[0]))
            ax[1].axis('off')
            ax[1].set_title('t = %d' % t)
            ax[2].imshow(tensor2uint(observation))
            ax[2].axis('off')
            ax[2].set_title('Observation')
            plt.show()
    weights = torch.softmax(log_weights, dim=0)
    ess_list.append(1/(weights**2).sum().item())
    idx = torch.multinomial(weights, sample_size, replacement=True)
    if return_auxiliary:
        return clean_output(x_t[idx]), ess_list#, memory_samples, ratio
    return clean_output(x_t[idx])



# DPS with DDPM and intrinsic scale
def dps_sampling_batch(model, nsamples, observation, forward_log_likelihood, scheduler, verbose = False, eta=1.0,time_rescaling=False, display_intermediary=False, batch_size=6):
    sample_size = model.config.sample_size
    for param in model.parameters():
        param.requires_grad = False
    # Init random noise
    x_t = torch.randn((nsamples, 3, sample_size, sample_size)).to(device)
    count = 0
    length = len(scheduler.timesteps)
    print_every = length // 20
    for t in tqdm.tqdm(scheduler.timesteps, disable=not verbose):
        alpha_t, beta_t, alpha_prod, next_alpha_prod = alpha_beta(scheduler, t)
        # Guidance weight
        # eta_t = ...
        eta_t =  beta_t / torch.sqrt(alpha_t)
        # Predict noisy residual eps_theta(x_t)
        for i in range(0, nsamples, batch_size):
            xtemp = x_t[i:min(i+batch_size, nsamples)].clone().detach().requires_grad_()
            epsilon_t = model(xtemp, t).sample

            # Get x0_hat and unconditional 
            # x_{t-1} = a_t * x_t + b_t * epsilon(x_t) + sigma_t z_t
            # with b_t = eta_t
            predict = scheduler.step(epsilon_t, t, xtemp) 
            x0_hat  = clean_output(predict.pred_original_sample)
            x_prev  = predict.prev_sample

            # Guidance
            rescaling_factor = time_rescaling * (1-alpha_prod)/eta
            g = - forward_log_likelihood(x0_hat, observation, rescaling_factor)
            guidance = torch.autograd.grad(g.sum(), xtemp)[0]
            # DPS update rule = DDPM update rule + guidance
            with torch.no_grad():
                x_t[i:min(i+batch_size, nsamples)] = x_prev - eta_t * guidance
                x_t = x_t.detach_()

        if count % print_every == 0 and display_intermediary:
            # Show progress of the sampling
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(tensor2uint(x_t[0]))
            ax[0].axis('off')
            ax[0].set_title('t = %d' % t)
            ax[1].imshow(tensor2uint(x0_hat[0]))
            ax[1].axis('off')
            ax[1].set_title('t = %d' % t)
            plt.show()
        count += 1

    return clean_output(x_t)


# DPS with DDPM and intrinsic scale
def dps_sampling(model, nsamples, observation, forward_log_likelihood, scheduler, verbose = False, eta=1.0,time_rescaling=False, display_intermediary=False):
    sample_size = model.config.sample_size
    for param in model.parameters():
        param.requires_grad = False
    # Init random noise
    x_T = torch.randn((nsamples, 3, sample_size, sample_size)).to(device)
    x_t = x_T
    count = 0
    length = len(scheduler.timesteps)
    print_every = length // 20
    for t in tqdm.tqdm(scheduler.timesteps, disable=not verbose):
        
        # Predict noisy residual eps_theta(x_t)
        x_t.requires_grad_()
        epsilon_t = model(x_t, t).sample

        # Get x0_hat and unconditional 
        # x_{t-1} = a_t * x_t + b_t * epsilon(x_t) + sigma_t z_t
        # with b_t = eta_t
        predict = scheduler.step(epsilon_t, t, x_t) 
        x0_hat  = clean_output(predict.pred_original_sample)
        x_prev  = predict.prev_sample # unconditional DDPM sample x_{t-1}'
        alpha_t, beta_t, alpha_prod, next_alpha_prod = alpha_beta(scheduler, t)
        # Guidance
        #f = torch.norm(forward_model(x0_hat) - y)**2/2/(sigma_y**2+(1-alpha_prod)/eta)
        #g = torch.autograd.grad(f, x_t)[0]
        rescaling_factor = time_rescaling * (1-alpha_prod)/eta
        quadratic, constant, log_factor = forward_log_likelihood(x0_hat, observation, rescaling_factor)
        g = -quadratic/constant - log_factor
        guidance = torch.autograd.grad(g.sum(), x_t)[0]
        # Guidance weight
        # eta_t = ...
        eta_t =  beta_t / torch.sqrt(alpha_t)

        # DPS update rule = DDPM update rule + guidance
        x_t = x_prev - eta_t * guidance
        x_t = x_t.detach_()

        if count % print_every == 0 and display_intermediary:
            # Show progress of the sampling
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(tensor2uint(x_t[0]))
            ax[0].axis('off')
            ax[0].set_title('t = %d' % t)
            ax[1].imshow(tensor2uint(x0_hat[0]))
            ax[1].axis('off')
            ax[1].set_title('t = %d' % t)
            plt.show()
        count += 1

    return clean_output(x_t)
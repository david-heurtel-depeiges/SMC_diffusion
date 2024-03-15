import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import abc
import numpy as np
import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExactScoreNetwork(nn.Module):
    def __init__(self,centers, stds, weights, beta_min = 0.1, beta_max = 20.0, t_min = 1e-4, t_max = 1.0, beta_schedule = 'linear'):
        super(ExactScoreNetwork, self).__init__()
        self.centers = centers.to(device)
        self.stds = stds.to(device)
        self.weights = weights.to(device)
        # Time parameters same as in default ContinuousVPSDE
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_min = t_min
        self.t_max = t_max
        self.beta_schedule = beta_schedule
        self.ndim = centers.shape[1]
        self.n_centers = centers.shape[0]
        self.config = {'centers': centers, 'stds': stds, 'weights': weights, 'beta_min': beta_min, 'beta_max': beta_max, 't_min': t_min, 't_max': t_max, 'beta_schedule': beta_schedule}
        if self.t_max < self.t_min:
            raise ValueError('t_max must be greater than t_min')
        if self.t_max != 1.0:
            raise NotImplementedError('t_max != 1.0 is not implemented yet and behavior is not guaranteed')
        self.beta_schedule = beta_schedule
        if beta_schedule == 'linear':
            self.beta = lambda t: (self.beta_min +  (self.beta_max - self.beta_min) * t)
            self.Beta = lambda t: (self.beta_min * t + 1/2*(self.beta_max - self.beta_min) * t**2)
        elif beta_schedule == 'cosine':
            self.beta = lambda t: self.beta_min + (self.beta_max - self.beta_min)*(1 - torch.cos(np.pi*t))/(2)
            self.Beta = lambda t: self.beta_min * t + (self.beta_max - self.beta_min)*(t - torch.sin(np.pi*t)/np.pi)/2

    def log_prob(self, x, t):
        # x: (n_samples, ndim)
        # t: (n_samples, 1)
        # returns: (n_samples, 1)
        centers_t = self.centers.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # Forward the centers through the SDE
        Beta_t = self.Beta(t).reshape(-1, 1, 1)
        centers_t = torch.exp(-Beta_t/2)*centers_t
        # Forward the stds through the SDE
        stds_t = self.stds.unsqueeze(0).repeat(x.shape[0], 1,1)
        stds_t_2 = torch.exp(-Beta_t)*stds_t**2 + 1 - torch.exp(-Beta_t) # (n_samples, n_centers, ndim)
        # Weights are not time dependent
        weights_t = self.weights.unsqueeze(0).repeat(x.shape[0], 1)
        x_t = x.unsqueeze(1).repeat(1, self.n_centers, 1)
        log_prob = (-((x_t - centers_t)**2)/(2*stds_t_2)).sum(-1)-self.ndim/2*torch.log(2*np.pi*stds_t_2).sum(-1)
        log_prob = log_prob + torch.log(weights_t)
        return torch.logsumexp(log_prob, dim=1)
    
    def forward(self, x, t):
        # x: (n_samples, ndim)
        # t: (n_samples, 1)
        # returns: (n_samples, ndim)
        x.requires_grad_(True)
        log_prob = self.log_prob(x, t)
        grad_log_prob = torch.autograd.grad(log_prob.sum(), x, create_graph=True)[0]
        Beta_t = self.Beta(t).reshape(-1, 1)
        return - torch.sqrt(1 - torch.exp(-Beta_t)) * grad_log_prob   

class ContinuousSDE(abc.ABC):
    def __init__(self):
        super(ContinuousSDE, self).__init__()
    @abc.abstractmethod
    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the SDE
        pass
    @abc.abstractmethod
    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the noise terms
        pass
    @abc.abstractmethod
    def reverse(self, x, t, modified_score):
        ## Returns the drift and brownian term of the reverse SDE for a given modified score
        pass
    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass
    @abc.abstractmethod
    def prior_log_likelihood(self, z):
        pass
    @abc.abstractmethod
    def rescale_additive_to_preserved(self, x, t):
        pass
    @abc.abstractmethod
    def rescale_preserved_to_additive(self, x, t):
        pass
    @abc.abstractmethod
    def noise_level(self, t):
        pass
    @abc.abstractmethod
    def tweedie_reverse(self, x, t, modified_score):
        pass
    @abc.abstractmethod
    def ode_drift(self, x, t, modified_score):
        ## Used for the ODE solver as well as likelihood computations
        pass

class ExactContinuousVPSDE(ContinuousSDE):
    def __init__(self, beta_min = 0.1, beta_max = 20.0, t_min = 1e-4, t_max = 1.0, beta_schedule = 'linear'):
        super(ExactContinuousVPSDE, self).__init__()
        self.beta_0 = beta_min
        self.beta_T = beta_max
        self.tmin = t_min
        self.tmax = t_max
        if self.tmax < self.tmin:
            raise ValueError('t_max must be greater than t_min')
        if self.tmax != 1.0:
            raise NotImplementedError('t_max != 1.0 is not implemented yet and behavior is not guaranteed')
        self.beta_schedule = beta_schedule
        if beta_schedule == 'linear':
            self.beta = lambda t: (self.beta_0 +  (self.beta_T - self.beta_0) * t)
            self.Beta = lambda t: (self.beta_0 * t + 1/2*(self.beta_T - self.beta_0) * t**2)
        elif beta_schedule == 'cosine':
            self.beta = lambda t: self.beta_0 + (self.beta_T - self.beta_0)*(1 - torch.cos(np.pi*t))/(2)
            self.Beta = lambda t: self.beta_0 * t + (self.beta_T - self.beta_0)*(t - torch.sin(np.pi*t)/np.pi)/2

    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the forward SDE
        beta_t = self.beta(t).reshape(-1, 1)
        drift = (- beta_t/2)* x_t
        brownian = torch.sqrt(beta_t)*torch.randn_like(x_t)
        return drift, brownian
    
    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the rescaled noise terms
        Beta_t = self.Beta(t).reshape(-1, 1)
        mean = torch.exp(-Beta_t/2)*x
        seed = torch.randn_like(x)
        noise = torch.sqrt(1 - torch.exp(-Beta_t))*seed
        x_tilde = mean + noise
        return x_tilde, mean, seed

    def reverse(self, x, t, modified_score):
        beta_t = self.beta(t).reshape(-1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta(t))).reshape(-1, 1)
        drift = -(beta_t/2 ) * x + (beta_t/sq_1_expB_t)*modified_score 
        brownian = torch.sqrt(beta_t)*torch.randn_like(x)
        return drift, brownian

    def tweedie_reverse(self, x, t, modified_score):
        t = torch.clamp(t, self.tmin, self.tmax)
        Beta_t = self.Beta(t).reshape(-1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-Beta_t))
        sq_expBt = torch.exp(-Beta_t/2)
        return (x - sq_1_expB_t*modified_score)/sq_expBt

    def ode_drift(self, x, t, modified_score):
        beta_t = self.beta(t).reshape(-1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta(t))).reshape(-1, 1)
        drift = (beta_t/2 ) * x - (1/2 * beta_t/sq_1_expB_t)*modified_score 
        return drift

    def prior_sampling(self, shape):
        return torch.randn(shape).to(device)

    def prior_log_likelihood(self, z):
        return 1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z**2, dim=[1, 2, 3]))
    
    def rescale_additive_to_preserved(self, x, t):
        return x * torch.exp(-self.Beta(t)/2).reshape(-1, 1)

    def rescale_preserved_to_additive(self, x, t):
        return x / torch.exp(-self.Beta(t)/2).reshape(-1, 1)

    def noise_level(self, t):
        return torch.sqrt(1 - torch.exp(-self.Beta(t)))/torch.exp(-self.Beta(t)/2)
    
    def get_closest_timestep(self, noise_level, n_step_method = 20, method = 'newton'):
        ## Solves the equation noise_level = self.noise_level(t) for t
        ## This is done analytically, noting that noise_level = sqrt(1 - exp(-Beta))/exp(-Beta/2)
        ## We first substitute x = exp(-Beta/2) and solve for x, then we solve for t (simple polynomial)
        if self.beta_schedule == 'linear':
            delta = self.beta_0**2 + 2*(self.beta_T - self.beta_0)*torch.log(1+noise_level**2)
            timesteps = (-self.beta_0 + torch.sqrt(delta))/(self.beta_T - self.beta_0)
            return torch.clamp(timesteps, self.tmin, self.tmax) ## TODO: check if this is correct
        elif self.beta_schedule == 'cosine':
            if type(noise_level) == float or type(noise_level) == int:
                noise_level = torch.tensor(noise_level)
            t_guess = 1/2*torch.ones_like(noise_level)
            if method == 'implicit':
                def implicit_func(t):
                    return (torch.log(1+noise_level**2)+(self.beta_T - self.beta_0)*torch.sin(np.pi*t)/(2*np.pi))*(2/(self.beta_T + self.beta_0))
                ## Solve implicit_func(t) = t
                timesteps = t_guess
                for _ in range(n_step_method):
                    timesteps = implicit_func(timesteps)
            elif method == 'newton':
                def newton_func(t):
                    return (self.beta_T + self.beta_0)/2*t - (torch.log(1+noise_level**2)+(self.beta_T - self.beta_0)*torch.sin(np.pi*t)/(2*np.pi))
                def newton_func_prime(t):
                    return (self.beta_T + self.beta_0)/2 - (self.beta_T - self.beta_0)*torch.cos(np.pi*t)/2
                timesteps = t_guess
                for _ in range(n_step_method):
                    timesteps = timesteps - newton_func(timesteps)/newton_func_prime(timesteps)
            return torch.clamp(timesteps, self.tmin, self.tmax)
        else:
            raise NotImplementedError('The beta schedule {} is not implemented'.format(self.beta_schedule))
        
class DiffusionModel(nn.Module):
    def __init__(self, sde, network):
        super(DiffusionModel, self).__init__()
        self.sde = sde
        self.network = network

    def loss(self, batch):
        raise NotImplementedError

    def generate_image(self, sample_size, channel, size, sample=None, initial_timestep=None):
        raise NotImplementedError

    def ddim(self, sample_size, channel, size, schedule):
        raise NotImplementedError
    
def get_schedule(schedule_type, **kwargs):
    """
    Returns a schedule of time steps for a differential equation solver (SDE or ODE)
    Args:
        schedule_type: str, type of schedule
        **kwargs: additional arguments for the schedule
    Returns:
        schedule: torch.Tensor, schedule of time steps
    """
    if schedule_type == 'linear':
        return linear_schedule(**kwargs)
    elif schedule_type == 'power_law':
        return power_law_schedule(**kwargs)
    else:
        raise NotImplementedError

def linear_schedule(**kwargs):
    """
    Returns a linear schedule of time steps for a differential equation solver (SDE or ODE)
    **kwargs should contain the following keys:
        t_min: float, minimum time step
        t_max: float, maximum time step
        n_iter: int, number of time steps
    Args:
        **kwargs: additional arguments for the schedule
    Returns:
        schedule: torch.Tensor, schedule of time steps
        """
    ## TODO parrallelize this
    if 't_min' not in kwargs:
        kwargs['t_min'] = 1e-4
    if 't_max' not in kwargs:
        kwargs['t_max'] = 1
    if 'n_iter' not in kwargs:
        kwargs['n_iter'] = 1000
    if type(kwargs['t_min']) == torch.Tensor:
        assert type(kwargs['t_max']) == torch.Tensor
        return torch.linspace(0,1, kwargs['n_iter']).to(kwargs['t_min'].device).reshape(1,-1) * (kwargs['t_max'] - kwargs['t_min']).unsqueeze(-1) + kwargs['t_min'].unsqueeze(-1)
    return torch.linspace(kwargs['t_min'], kwargs['t_max'], kwargs['n_iter'])

def power_law_schedule(**kwargs):
    """
    Returns a power law schedule of time steps for a differential equation solver (SDE or ODE)
    **kwargs should contain the following keys:
        t_min: float, minimum time step
        t_max: float, maximum time step
        n_iter: int, number of time steps
        power: float, power law exponent
    Args:
        **kwargs: additional arguments for the schedule
    Returns:
        schedule: torch.Tensor, schedule of time steps
    """
    ## TODO parrallelize this
    if 't_min' not in kwargs:
        kwargs['t_min'] = 1e-4
    if 't_max' not in kwargs:
        kwargs['t_max'] = 1
    if 'n_iter' not in kwargs:
        kwargs['n_iter'] = 1000
    if 'power' not in kwargs:
        kwargs['power'] = 1.1
    n_iter = kwargs['n_iter']
    if type(kwargs['t_min']) == torch.Tensor:
        assert type(kwargs['t_max']) == torch.Tensor
        return (torch.linspace(0,1,n_iter)**kwargs['power']).to(kwargs['t_min'].device).reshape(1,-1)* (kwargs['t_max'] - kwargs['t_min']).unsqueeze(-1) + kwargs['t_min'].unsqueeze(-1)
    return torch.linspace(0,1,n_iter)**kwargs['power'] * (kwargs['t_max'] - kwargs['t_min']) + kwargs['t_min']

class EulerMaruyama():
    """
    Euler-Maruyama method for solving stochastic differential equations
    """
    def __init__(self, schedule):
        self.schedule = schedule

    def forward(self, x_init, f, gdW, reverse_time = True, verbose = False):
        """
        Solves the SDE dx = f(x, t) dt + g(x, t) dW_t
        Args:
            x_init: torch.Tensor, initial value of x
            f: function, drift term
            gdW: function, diffusion term (should also contain the random part of the diffusion term, will only be multiplied by sqrt(dt) in the solver)
            reverse_time: bool, whether to reverse the time schedule (used for backward SDEs)
            verbose: bool, whether to print progress bar
        Returns:
            x: torch.Tensor, a sample from the SDE at the final time step
        """
        if reverse_time:
            times = self.schedule
            times = times.flip(1)
        else:
            times = self.schedule
        x = x_init

        progress_bar = tqdm.tqdm(total = times.shape[1], disable=not verbose)
        for i in range(times.shape[1]-1):
            dt = (times[:,i+1] - times[:,i]).reshape(-1,1)
            timesteps = times[:,i].to(x.device).unsqueeze(1)
            x = x + f(x, timesteps) * dt + gdW(x, timesteps) * torch.sqrt(torch.abs(dt))
            if x.isnan().any():
                print("Nan encountered at time step {}".format(i))
                if f(x, timesteps).isnan().any():
                    print("Nan in drift")
                if gdW(x, timesteps).isnan().any():
                    print("Nan in diffusion")
                break
            x = x.detach()
            progress_bar.update(1)
        if reverse_time:
            ## flip back to the original time order (to avoid side effects)
            times = x.flip(1)
        return x
    

class ExactContinuousSBM(DiffusionModel):
    def __init__(self, sde, network):
        super(ExactContinuousSBM, self).__init__(sde, network)
        self.sde = sde
        self.network = network
        self.tmin = self.sde.tmin
        self.tmax = self.sde.tmax

    def loss(self, batch):
        timesteps = (torch.rand(batch.shape[0],1).to(device) * (self.tmax - self.tmin) + self.tmin)
        batch_tilde, _ , rescaled_noise = self.sde.sampling(batch, timesteps)
        rescaled_noise_pred = self.network(batch_tilde, timesteps)
        return F.mse_loss(rescaled_noise_pred, rescaled_noise)
        
    def generate_image(self, sample_size, sample=None, initial_timestep=None, verbose=False, schedule = None, solver = None):
        self.eval()
        if schedule is None:
            if initial_timestep is None:
                t_min = torch.tensor([self.sde.tmin]).repeat(sample_size).to(device)
                t_max = torch.tensor([self.sde.tmax]).repeat(sample_size).to(device)
            else:
                t_min = torch.tensor([self.sde.tmin]).repeat(sample_size).to(device)
                t_max = torch.tensor([initial_timestep]).repeat(sample_size).to(device)
            schedule = get_schedule('linear', t_min = t_min, t_max = t_max, n_iter = 1000)
        
        if solver is None:
            solver = EulerMaruyama(schedule)

        if sample is None:
            ndim = self.network.ndim
            sample = self.sde.prior_sampling((sample_size, ndim)).to(device)
        ## TODO more efficient for f, g? -> ok for our solver but not for others 
        def f(x_t, t):
            model_output = self.network(x_t, t)
            return self.sde.reverse(x_t, t, model_output)[0]
        def gdW(x_t, t):
            dummy_output = torch.zeros_like(x_t)
            return self.sde.reverse(x_t, t, dummy_output)[1]
        if type(self.network) == ExactScoreNetwork:
                gen = solver.forward(sample, f, gdW, reverse_time = True, verbose = verbose)
        else:
            with torch.no_grad():
                gen = solver.forward(sample, f, gdW, reverse_time = True, verbose = verbose)
        return gen

    def ode_sampling(self, sample_size, sample = None, initial_timestep = None, verbose=True): ## TODO scheduler maybe only in continuous time? 
        self.eval()
        raise NotImplementedError("Not implemented yet")
        channel, size = self.network.in_c, self.network.sizes[0]
        if initial_timestep is None:
            tot_steps = self.sde.N
        else:
            tot_steps = initial_timestep
        with torch.no_grad():
            timesteps = list(range(tot_steps))[::-1]
            if sample is None:
                sample = self.sde.prior_sampling((sample_size, channel, size, size))
            progress_bar = tqdm.tqdm(total=tot_steps, disable=not verbose)
            for t in timesteps:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                residual = self.network(sample, time_tensor)
                sample = self.ode_step(residual, time_tensor[0], sample)
                progress_bar.update(1)
            progress_bar.close()

        return sample
    
    def log_likelihood(self, batch, initial_timestep = None, verbose=True, repeat = 1):
        '''Sample in forward time the ODE and compute the log likelihood of the batch, see [REF]'''
        self.eval()
        raise NotImplementedError("Not implemented yet")
        log_likelihood = torch.zeros(len(batch)).to(device)
        with torch.no_grad():
            N = self.sde.N
            progress_bar = tqdm.tqdm(total=N, disable=not verbose)
            for i in range(N):
                timesteps = torch.tensor([i]).repeat(len(batch)).to(device)
                modified_score = self.network(gen, timesteps)
                gen -= self.sde.ode_drift(gen, timesteps, modified_score)
                progress_bar.update(1)
                log_likelihood_increase = 0
                with torch.enable_grad():
                    for _ in range(repeat):
                        epsilon = torch.randn_like(batch)
                        gen.requires_grad = True
                        reduced = torch.sum(modified_score * epsilon)
                        grad = torch.autograd.grad(reduced, gen, create_graph=True)[0]
                        gen.requires_grad = False
                        log_likelihood_increase += torch.sum(grad * epsilon, dim=(1, 2, 3))
                        ## TODO add zero grad
                log_likelihood_increase /= repeat
                log_likelihood += log_likelihood_increase/(N-initial_timestep)
            progress_bar.close()
            log_likelihood += self.sde.prior_log_likelihood(batch)
        self.train()
        return log_likelihood
import torch
import torch.nn as nn
from common.schedules import get_cosine_schedule, get_linear_schedule
from common.utils import *

class Diffusion(nn.Module):
    def __init__(self,
                 input_shape : tuple[int, ...],
                 model : nn.Module,
                 ts : int = 1000,
                 diffusion_type : str = 'ddpm',
                 sched_type : str='cosine'):
        super().__init__()
        
        self.input_shape = input_shape
        self.model = model
        self.ts = ts
        self.diffusion_type = diffusion_type
        
        # build variance schedule
        if sched_type == 'cosine':
            alpha_t = get_cosine_schedule(ts)
        elif sched_type == 'linear':
            alpha_t = get_linear_schedule(ts)
        else:
            raise NotImplementedError
        
        # adjust alpha_t dimensions to match input
        self.alpha_t = unsqueeze_to(alpha_t, len(self.input_shape) + 1)
        self.sigma_t = (1 - alpha_t ** 2).clamp(min=0) ** 0.5

        self.register_buffer("alpha_t", self.alpha_t)
        self.register_buffer("sigma_t", self.sigma_t)
        
    def loss(self, target):
        batch_size = target.shape[0]

        # sample t uniformly
        t_sample = torch.randint(1, self.ts + 1, size=(batch_size, ), device = target.device)
        eps = torch.randn_like(target)

        # forward diffusion of q(x_t | x_{t-1}) w reparameterisation trick
        x_t = self.alpha_t[t_sample] * target + self.sigma_t[t_sample] * eps

        pred = self.model(x_t, t_sample)

        # eps matching
        return 0.5 * (eps - pred) ** 2

    def sample(self, gaussian : torch.Tensor, diffuse_steps : int):
        batch_size = gaussian.shape[0]

        t_start = torch.empty((batch_size,), device=gaussian.device)
        t_end = torch.empty((batch_size,), device=gaussian.device)

        subseq = torch.linspace(self.ts, 0, self.ts + 1).round()

        samples = []
        samples.append(gaussian)

        for i, (t_start, t_end) in enumerate(zip(subseq[:-1], subseq[1:])):
            print(f"starting step {i}")

            x_t = samples[-1]

            # slowly step through markov process
            t_start.fill_(t_start)
            t_end.fill_(t_end)

            # when t == 0, we add zero noise to ensure distribution matches.
            # this pushes the learning of the reconstruction term.
            eps = 0 if t_end == 0 else torch.randn_like(gaussian)

            # select gamma / noise
            if self.diffusion_type == 'ddim':
                gamma_t = 0.
            elif self.diffusion_type == 'ddpm':
                gamma_t = self.sigma_t[t_end] / self.sigma_t[t_start] * \
                    (1 - self.alpha_t[t_start] ** 2 / self.alpha_t[t_end] ** 2) ** 0.5
            else:
                raise NotImplementedError
            
            pred_eps = self.model(x_t, t_start)
            x_0 = (x_t - self.sigma_t[t_start] * pred_eps) / self.sigma_t[t_start]
            x_t = (self.alpha_t[t_end] * x_0) \
                + (self.sigma_t[t_end] ** 2 - gamma_t ** 2).clamp(min=0.) ** 0.5 * pred_eps \
                    + (gamma_t * eps)
            
            samples.append(x_t)

        return samples


            


    


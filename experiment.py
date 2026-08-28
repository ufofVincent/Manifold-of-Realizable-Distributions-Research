import torch 
import torch.nn as nn 
import torch.optim as optim
import copy
from collections.abc import Callable
import math
import matplotlib.pyplot as plt
import numpy as np




class DistributionGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num, hidden_layer_dim):
        super(DistributionGenerator, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_layer_dim))
        layers.append(nn.Sigmoid())
        
        self.layer_num = hidden_layer_num + 2
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        for _ in range(hidden_layer_num - 1):
            layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            layers.append(nn.Sigmoid())
            
        layers.append(nn.Linear(hidden_layer_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, input: torch.tensor):
        
        return self.layers(input)
        
    def compute_NTK(self, x: torch.tensor, y:torch.tensor):
        
        jacobians_1 = []
        jacobians_2 = []
        
        y_1 = self.forward(x)
        y_2 = self.forward(y)
        
        
        for _ in range (self.output_dim):
            gradient_1 = torch.autograd.grad(
                outputs= y_1[_],
                inputs = list(self.parameters()),
                create_graph= True,
                retain_graph= True
            )
                                    
            jacobians_1.append(torch.cat([g.flatten() for g in gradient_1]))
            
            
        for _ in range(self.output_dim):              
            gradient_2 = torch.autograd.grad(
                outputs= y_2[_],
                inputs = list(self.parameters()),
                create_graph= True,
                retain_graph= True
            )
            
            jacobians_2.append(torch.cat([g.flatten() for g in gradient_2]))
            
            
        jacobian_1 = torch.stack(jacobians_1)
        jacobian_2 = torch.stack(jacobians_2)                            
        
        return jacobian_1 @ jacobian_2.T
    

def make_swiss_roll(n: int, output_dim: int, noise: float = 0.1):
    assert output_dim >= 3

    t = 1.5 * math.pi * (1 + 2 * torch.rand(n))
    h = 6.0 * torch.rand(n)

    data = torch.zeros(n, output_dim)
    data[:, 0] = t * torch.cos(t)
    data[:, 1] = h
    data[:, 2] = t * torch.sin(t)

    data += noise * torch.randn_like(data)
        
    return data

def sample_from_distribution(n: int, distribution, output_dim):
    
    data = distribution(n, output_dim)
    
    idx = torch.randint(0, n, ()).item()
    return data[idx]    


    
def compute_drift(input_sample: torch.tensor, generator: DistributionGenerator, kernel: Callable[[torch.tensor, torch.tensor], float], sample_num: int, sampler):

    total_drift = 0.0
    
    z_q = 0.0
    z_p = 0.0
    
    x = generator(input_sample)

    for _ in range(sample_num):
        
        
        eps = torch.randn(generator.input_dim)
        y_minus = generator(eps)
        
        y_plus = sampler(sample_num, make_swiss_roll, generator.output_dim)
                
        total_drift += kernel(x, y_minus) * kernel(x, y_plus) * (y_plus - y_minus)
        
        # also empirically compute the normalization consatnts
        z_p += kernel(x, y_plus)
        z_q += kernel(x, y_minus)
        
    z_q = z_q / sample_num
    z_p = z_p /sample_num
    
    total_drift = total_drift / sample_num
    
    total_drift = total_drift / (z_p * z_q)
    
    
    return total_drift
        
def iterate(model: DistributionGenerator, sample_num: int, kernel_function: Callable[[torch.tensor, torch.tensor], float], optimizer: torch.optim, sampler):
    
    
    old_model = copy.deepcopy(model)
    
    #optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    optimizer.zero_grad()

    total_loss = 0.0
    
    for _ in range(sample_num):
        sample = torch.randn(model.input_dim)
        
        with torch.no_grad():
            frozen_target = model(sample) + compute_drift(sample, model, kernel_function, 100, sampler)
        
        total_loss += torch.linalg.norm(model(sample) - frozen_target) ** 2 
        
    loss = total_loss / sample_num
    
    
    loss.backward()
    
    optimizer.step()
    
    return model, old_model

# experiment: for each iteration, numerically approximate NTK-induced metric 

def gaussian_kernel(x: torch.tensor, y: torch.tensor) -> float:

    return torch.exp(-torch.linalg.norm(x-y)).item()

if __name__ == "__main__":
    
    torch.set_default_dtype(torch.float64)

    
    model = DistributionGenerator(3, 5, 20, 10)
        
    
    total_training_step = 50
    
    n_samples = 30
    
    samples = [torch.randn(model.input_dim) for i in range (n_samples)]
    
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=0)
    
            
    

        
    # compute ntk 
    d = model.output_dim

    for i in range(n_samples):
        for j in range(n_samples):
            sample_matrix = model.compute_NTK(
                samples[i],
                samples[j],
            )

            ntk_gram = torch.zeros(n_samples * model.output_dim, n_samples * model.output_dim)

            ntk_gram[
                i * d:(i + 1) * d,
                j * d:(j + 1) * d,
            ] = sample_matrix.detach()                        
        
    ntk_gram = (ntk_gram + ntk_gram.T) / 2

    inverse_ntk = torch.linalg.pinv(
        ntk_gram,
        hermitian=True,
        rtol=1e-10
    )
        
    metrics = []
    parameter_perturbations = []
    
    for _ in range(total_training_step):
        
        t_next, t_curr = iterate(model, 50, gaussian_kernel, optimizer, sample_from_distribution)
        
        
        # first, compute the parameter perturbation between the next model and the first
        
        # extract out all parameters then take L2 norm
        
        
        t_next_parameters = list(t_next.parameters())
        
        t_next_parameters_new = torch.cat([p.detach().flatten() for p in t_next.parameters()])

        t_curr_parameters_new = torch.cat([p.detach().flatten() for p in t_curr.parameters()])
                        
            
        
        perturbation = torch.sum((t_next_parameters_new - t_curr_parameters_new) ** 2)
        
        parameter_perturbations.append(perturbation.item())
        
        ## numerically approximate velocity field u_t ≈ f_t+1 - f_t
                
        metric = 0.0
        
        stacked_velocities = torch.empty(0)
        
        for i in range (n_samples):
            
            epsilon = samples[i]
            
            u_t_1 = t_next(epsilon) - t_curr(epsilon)
            
            stacked_velocities = torch.cat([stacked_velocities, u_t_1])
                
            
        metric = stacked_velocities @ inverse_ntk @ stacked_velocities
                
        
            
        #metric = metric / (n_samples ** 2)
        
        metrics.append(metric.item())
                
        # now compute L_2 norm of error
                
        

        print(f"At step {_}, NTK metric is approximately {metric}. The parameter perturbation L2 norm is {perturbation}")
        
        
        model = t_next
        
        #recompute NTK
        
        for i in range(n_samples):
            for j in range(n_samples):
                sample_matrix = model.compute_NTK(
                    samples[i],
                    samples[j],
                )

                ntk_gram[
                    i * d:(i + 1) * d,
                    j * d:(j + 1) * d,
                ] = sample_matrix.detach()                        
        
        ntk_gram = (ntk_gram + ntk_gram.T) / 2

        inverse_ntk = torch.linalg.pinv(
            ntk_gram,
            hermitian=True,
            rtol=1e-10
        )

    print("Metrics for this run", metrics)
    print("Parameter perturbation L2 norms for this run:", parameter_perturbations)
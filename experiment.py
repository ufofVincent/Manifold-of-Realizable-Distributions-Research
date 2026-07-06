import torch 
import torch.nn as nn 
import torch.optim as optim
import copy
from collections.abc import Callable


class DistributionGenerator(torch.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num, hidden_layer_dim):
        super(self).__init__()
        
        self.layers = []
        
        self.layers.append(nn.linear(input_dim, hidden_layer_dim))
        
        self.layer_num = hidden_layer_num + 2
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        for _ in range(hidden_layer_num):
            self.layers.append(nn.linear(hidden_layer_dim, hidden_layer_dim))
            
        self.layers.append(nn.Linear(hidden_layer_dim, output_dim))
        
    def forward(self, input: torch.tensor):
        
        for _ in range(self.layer_num):
            
            input = self.layers[_](input)
            
        return input

class EmpiricalSampler:
    def __init__(self, data: torch.Tensor):
        self.data = data

    def sample(self, n: int):
        idx = torch.randint(0, len(self.data), (n,))
        return self.data[idx]

    
def compute_drift(input_sample: torch.tensor, generator: DistributionGenerator, kernel: Callable[[torch.tensor, torch.tensor], float], sample_num: int, empirical_sampler: EmpiricalSampler):

    total_drift = 0.0
    
    z_q, z_p = 0.0

    for _ in range(sample_num):
        
        
        eps = torch.rand(generator.input_dim)
        y_minus = generator(eps)
        
        x = generator(input_sample)
        
        y_plus = empirical_sampler.sample()
        
        total_drift += kernel(x, y_minus) * kernel(x, y_plus) * (y_plus - y_minus)
        
        z_p += kernel(x, y_plus)
        z_q += kernel(x, y_minus)
        
    z_q = z_q / sample_num
    z_p = z_p /sample_num
        
    total_drift = total_drift / sample_num
    
    return total_drift / (z_p * z_q)
        
def iterate(model: DistributionGenerator, sample_num: int, drift_field, kernel_function):
    
    old_model = copy.deepcopy(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    optimizer.zero_grad()

    total_loss = 0.0
    
    for _ in range(sample_num):
        sample = torch.randn(model.input_dim)
        
        with torch.no_grad:
            frozen_target = model(sample) + drift_field(sample, model, kernel_function, 100)
        
        total_loss += torch.linalg.norm(model(sample) - frozen_target) ** 2 
        
    loss = sum / sample_num
    
    
    loss.backward()
    
    optimizer.step()
    
    return model, old_model
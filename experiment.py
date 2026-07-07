import torch 
import torch.nn as nn 
import torch.optim as optim
import copy
from collections.abc import Callable


class DistributionGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num, hidden_layer_dim):
        super(DistributionGenerator, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_layer_dim))
        
        self.layer_num = hidden_layer_num + 2
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        for _ in range(hidden_layer_num):
            layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            
        layers.append(nn.Linear(hidden_layer_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, input: torch.tensor):
        
        return self.layers(input)
    
    def ping(self):
        print(list(self.parameters()))
    
    def compute_NTK(self, x: torch.tensor, y:torch.tensor):
        
        
        gradient_1 = torch.autograd.grad(
            outputs= self.forward(x),
            inputs = list(self.parameters()),
            create_graph= True,
            retain_graph= True
        )
        
        print(gradient_1)
        
        gradient_1 = gradient_1[0]
        
        gradient_2 = torch.autograd.grad(
            outputs= self.forward(y),
            inputs = list(self.parameters()),
            create_graph= True,
            retain_graph= True
        )[0]
        
        gradient_1_flattened = torch.cat([g.reshape(-1) for g in gradient_1])
        gradient_2_flattened= torch.cat([g.reshape(-1) for g in gradient_2])
        
        #print(gradient_1_flattened)
                
        return torch.outer(gradient_1_flattened,gradient_2_flattened)
    
model = DistributionGenerator(3, 1, 5, 10)
x = torch.randn(3)
y = torch.randn(3)
print(model.compute_NTK(x,y))



class EmpiricalSampler:
    def __init__(self, data: torch.Tensor):
        self.data = data

    def sample(self, n: int):
        idx = torch.randint(0, len(self.data), (n,))
        return self.data[idx]
    


    
def compute_drift(input_sample: torch.tensor, generator: DistributionGenerator, kernel: Callable[[torch.tensor, torch.tensor], float], sample_num: int, empirical_sampler: EmpiricalSampler):

    total_drift = 0.0
    
    z_q, z_p = 0.0
    
    x = generator(input_sample)

    for _ in range(sample_num):
        
        
        eps = torch.randn(generator.input_dim)
        y_minus = generator(eps)
        
        y_plus = empirical_sampler.sample()
        
        total_drift += kernel(x, y_minus) * kernel(x, y_plus) * (y_plus - y_minus)
        
        # also empirically compute the normalization consatnts
        z_p += kernel(x, y_plus)
        z_q += kernel(x, y_minus)
        
    z_q = z_q / sample_num
    z_p = z_p /sample_num
    
    total_drift = total_drift / sample_num
    
    return total_drift / (z_p * z_q)
        
def iterate(model: DistributionGenerator, sample_num: int, kernel_function: Callable[[torch.tensor, torch.tensor], float], optimizer: torch.optim):
    
    old_model = copy.deepcopy(model)
    
    #optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    optimizer.zero_grad()

    total_loss = 0.0
    
    for _ in range(sample_num):
        sample = torch.randn(model.input_dim)
        
        with torch.no_grad():
            frozen_target = model(sample) + compute_drift(sample, model, kernel_function, 100)
        
        total_loss += torch.linalg.norm(model(sample) - frozen_target) ** 2 
        
    loss = total_loss / sample_num
    
    
    loss.backward()
    
    optimizer.step()
    
    return model, old_model

# experiment: for each iteration, numerically approximate NTK-induced metric 

def gaussian_kernel(x: torch.tensor, y: torch.tensor) -> float:

    return torch.exp(-torch.linalg.norm(x-y)).item()

if __name__ == "__main__":
    model = DistributionGenerator(30, 100, 10, 50)
    
    total_training_step = 50
    
    n_squared_samples = 36
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    for _ in range(total_training_step):
        
        t_next, t_curr = iterate(model, 50, gaussian_kernel, optimizer)
        
        ## numerically approximate velocity field u_t ≈ f_t+1 - f_t
        
        u_t_1, u_t_2 = 0
        
        metric = 0.0
        
        for __ in range (n_squared_samples):
            
            epsilon = torch.randn(model.input_dim)
            epsilon_2 = torch.randn(model.input_dim)
            u_t_1 = t_next(epsilon) - t_curr(epsilon)
            
            u_t_2 = t_next(epsilon_2) - t_curr(epsilon_2)
            
            metric += u_t_1.T * model.compute_NTK(epsilon, epsilon_2) * u_t_2
            
        metric = metric / n_squared_samples
        
        
        print(f"At step {_}, NTK metric is approximately {metric}")
        
    

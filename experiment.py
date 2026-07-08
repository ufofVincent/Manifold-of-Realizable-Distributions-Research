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
        
        for _ in range(hidden_layer_num - 1):
            layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            
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
    



class EmpiricalSampler:
    def __init__(self, data: torch.Tensor):
        self.data = data

    def sample(self, n: int):
        idx = torch.randint(0, len(self.data), (n,))
        return self.data[idx]
    


    
def compute_drift(input_sample: torch.tensor, generator: DistributionGenerator, kernel: Callable[[torch.tensor, torch.tensor], float], sample_num: int, empirical_sampler: EmpiricalSampler):

    print("entered compute_drift()")

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
    
    print("entered iterate()")
    
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
    
    n_samples = 5
    
    samples = []
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    for i in range(n_samples):
        samples.append(torch.rand(model.input_dim))
        
    ntk_gram = torch.zeros(n_samples, n_samples)
    
    
    
    for i in range(n_samples):
        for j in range(n_samples):
            ntk_gram[i][j] = torch.trace(model.compute_NTK(samples[i],samples[j]))
            
    inverse_ntk = torch.inverse(ntk_gram)
    
    
    for _ in range(total_training_step):
        
        t_next, t_curr = iterate(model, 50, gaussian_kernel, optimizer)
        
        ## numerically approximate velocity field u_t ≈ f_t+1 - f_t
        
        u_t_1, u_t_2 = 0
        
        metric     
        for i in range (n_samples):
            for j in range(n_samples):
            
                epsilon = samples[i]
                epsilon_2 = samples[j]
                u_t_1 = t_next(epsilon) - t_curr(epsilon)
                
                u_t_2 = t_next(epsilon_2) - t_curr(epsilon_2)
                
                metric += torch.dot(u_t_1, u_t_2) * inverse_ntk[i,j]
                
        
            
        metric = metric / (n_samples ** 2)
        
        # now compute L_2 norm of error
        
        
        error = 0.0
        
        for sample in samples:
            error += torch.linalg.norm(t_next(sample) - t_curr(sample) - compute_drift(sample, model, gaussian_kernel, 50, sampler))
        
        error = error / n_samples
        
        

        print(f"At step {_}, NTK metric is approximately {metric}. The actual L2 error is {error}")
        
        model = t_next
        
    

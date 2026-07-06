import torch 
import torch.nn as nn 
import torch.optim as optim
import copy

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



def compute_drift(sample: torch.tensor, generator: DistributionGenerator, kernel_function, sample_num: int):
    
    #x = generator(sample)
    
    #return kernel_function(x, ) * kernel_function(x, )
    
    pass 
    
def iterate(model: DistributionGenerator, sample_num: int, drift_field, kernel_function):
    
    old_model = copy.deepcopy(model)
    
    for _ in range(sample_num):
        sample = torch.randn(model.input_dim)
        
        with torch.no_grad:
            frozen_target = model(sample) + drift_field(sample, model, kernel_function, 100)
        
        sum += torch.linalg.norm(model(sample) - frozen_target)
        
    loss = sum / sample_num
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    return model, old_model
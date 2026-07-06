import torch 
import torch.nn as nn 

class Generator(torch.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num, hidden_layer_dim):
        super(self).__init__()
        
        self.layers = []
        
        self.layers.append(nn.linear(input_dim, hidden_layer_dim))
        
        self.layer_num = hidden_layer_num + 2
        
        for _ in range(hidden_layer_num):
            self.layers.append(nn.linear(hidden_layer_dim, hidden_layer_dim))
            
        self.layers.append(nn.Linear(hidden_layer_dim, output_dim))
        
    def forward(self, input: torch.tensor):
        
        for _ in range(self.layer_num):
            
            input = self.layers[_](input)
            
        return input
    
    

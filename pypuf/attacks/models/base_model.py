import torch

'A basic model for attacks on XOR Arbiter PUFs'

class BasicModel(torch.nn.Module):
     def __init__(self, n_inputs, k):
        super(BasicModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(0, k):
            self.layers.append(torch.nn.Linear(n_inputs, 1, bias=False))
        
     def forward(self, x):
        y_pred = torch.ones(x.shape[0], 1).to(self.device)
            
        for layer in self.layers:
            y_pred = y_pred * layer(y_pred)
        return torch.tanh(y_pred)
    
     def predict(self, x):
        y_pred = 1
        for layer in self.layers:
            y_pred = y_pred * layer(y_pred)
        if torch.tanh(y_pred) >= 0:
            return 1
        else:
            return -1
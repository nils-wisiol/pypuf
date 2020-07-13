'''A basic model for attacks on XOR Arbiter PUFs'''
import torch

class BasicModel(torch.nn.Module):
    '''
    A Basic Attack model f using the tanh function for
    '''
    def __init__(self, n_inputs, k, device):
        super(BasicModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.device = device
        self.k = k
        self.n_inputs = n_inputs
        for i in range(0, k):
            self.layers.append(torch.nn.Linear(n_inputs, 1, bias=False))
        
    def forward(self, X):
        y_pred = torch.ones(X.shape[0], 1).to(self.device)
            
        for layer in self.layers:
            y_pred = y_pred * layer(X)
        return torch.tanh(y_pred)
    
    def predict(self, x):
        y_pred = 1
        for layer in self.layers:
            y_pred = y_pred * layer(x)
        if torch.tanh(y_pred) >= 0:
            return 1
        else:
            return -1
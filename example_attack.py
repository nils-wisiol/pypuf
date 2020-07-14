from pypuf.attacks.pipeline import pipeline
from pypuf.simulation.delay import XORArbiterPUF
from torch.optim import SGD
from torch.nn import MSELoss
import numpy as np

k = 3
input_size = 64

puf = XORArbiterPUF(n=input_size, k=k, noisiness=.1, seed=1)

model, losses, accuracy = pipeline(simulation_instance=puf, input_size=input_size, k=k, num_epochs=10000, device='cpu', random_seed=0)

print(accuracy)




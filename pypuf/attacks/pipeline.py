from ..io import random_inputs
from ..simulation.delay import LTFArray
import torch 
from sklearn.metrics import accuracy_score
from numpy.random import RandomState
import tqdm as tqdm_normal
from tqdm.notebook import tqdm as tqdm_notebook
from .models.base_model import BasicModel
from numpy import array, ones


def create_test_and_train(input_size, train_size=1000, test_size=100, k=2, random_seed=1234, device='cpu'):
    instance = LTFArray(ones(shape=(1, 4)), transform='id')
    X_train = random_inputs(n=input_size, N=train_size, seed=random_seed)
    y_train = instance.eval(X_train)
    X_test = random_inputs(n=input_size, N=test_size, seed=random_seed)
    y_test = instance.eval(X_test)

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    y_train = y_train.view(train_size, 1)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    return X_train, y_train, X_test, y_test


def train(model, X_train, y_train, criterion, optimizer, batch_size, notebook=False):
    if notebook:
        progress_bar = tqdm_notebook
    else:
        progress_bar = tqdm_normal

    losses = []

    for t in progress_bar(range(1000)):
        j = 0
        for i in range(0, len(X_train), batch_size):
            y_pred = model(X_train[j:i])
            loss = criterion(y_pred, y_train[j:i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j = i
    
        losses.append(loss.item())

    return losses

def pipeline(
    input_size=2000,
    k=64,
    random_seed=1234,
    device='cpu', 
    model=None,
    criterion=None,
    optimizer=None,
    batch_size=None,
    notebook=False):
    
    X_train, y_train, X_test, y_test = create_test_and_train(input_size, k, random_seed, device)

    model = BasicModel(input_size, k).to(device)

    losses = train(model, X_train, y_train, criterion, optimizer, batch_size, notebook)

    print(losses)

    predictions = []
    with torch.no_grad():
        for x in X_test:
            predictions.append(model.predict(x))
            #print()
            
            
    print(accuracy_score(y_test.cpu().numpy(), predictions))

pipeline()
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from random import shuffle
import random
from torchvision.models import resnet50
import os
import numpy as np

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def test(dataloader, model, loss_fn, device, acc_lst):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    acc_lst.append(correct)
    
def test_aug():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    data_dir = './data/fake/'
    tensors = os.listdir(data_dir)
    dataset1 = [(torch.load(data_dir + a_tensor), 0) for a_tensor in tensors]
    data_dir = './data/real/'
    tensors = os.listdir(data_dir)
    dataset2 = [(torch.load(data_dir + a_tensor), 1) for a_tensor in tensors]

    dataset = dataset1 + dataset2
    shuffle(dataset)
    training_data = dataset[:int(len(dataset) * 0.8)]
    test_data = dataset[int(len(dataset) * 0.8):]

    batch_size = 64

    # 데이터로더를 생성합니다.
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    acc_lst = []

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device, acc_lst)
    print(f"Done! The maximum accuracy was {max(acc_lst)}")
    return max(acc_lst)
    
if __name__ == '__main__':
    temp = test_aug()
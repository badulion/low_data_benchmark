from dynabench.dataset import DynabenchIterator, download_equation
from torch.utils.data import DataLoader
from dynabench.model.point._gcn import GCN
from dynabench.model._utils import PointIterativeWrapper

import torch.optim as optim
import torch.nn as nn

#download_equation('advection', structure='cloud', resolution='high')

advection_train_iterator = DynabenchIterator(split="train",
                                           equation='advection',
                                           structure='cloud',
                                           resolution='high',
                                           lookback=1,
                                           rollout=1)

train_loader = DataLoader(advection_train_iterator, batch_size=1, shuffle=True)

## width -> number of channels in convolution layers
## for an NxN grid -> max n_modes = [N//2 + 1, N//2 + 1], channels depends on equation
net = GCN(num_convs=3, in_features=1, hidden_features=32, dropout=0.1, knn=10)
model = PointIterativeWrapper(net)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    for i, (x, y, p) in enumerate(train_loader):
        x, y, p = x.float(), y.float(), p.float() # only use the first channel and convert to float32
        optimizer.zero_grad()
        y_pred = model(x, p)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

advection_test_iterator = DynabenchIterator(split="test",
                                          equation='advection',
                                          structure='cloud',
                                          resolution='low',
                                          lookback=1,
                                          rollout=16)

test_loader = DataLoader(advection_test_iterator, batch_size=1, shuffle=False)

model.eval()

loss_values = []
for i, (x, y, p) in enumerate(test_loader):
    x, y, p = x[:,0].float(), y.float(), p.float() # only use the first channel and convert to float32
    y_pred = model(x, p, t_eval=range(1,17))
    loss = criterion(y_pred, y)
    loss_values.append(loss.item())

print(f"Mean Loss: {sum(loss_values) / len(loss_values)}")
from dynabench.dataset import DynabenchIterator, download_equation
from torch.utils.data import DataLoader
from dynabench.model.point._point_net import PointNet
from dynabench.model._graph_wrapper import GraphIterativeWrapper

import torch.optim as optim
import torch.nn as nn
import torch

#download_equation('advection', structure='cloud', resolution='low')

advection_train_iterator = DynabenchIterator(split="train",
                                           equation='advection',
                                           structure='cloud',
                                           resolution='low',
                                           lookback=1,
                                           rollout=1)

train_loader = DataLoader(advection_train_iterator, batch_size=16, shuffle=True)

net = PointNet(input_size=1, 
                 lookback=1, 
                 intermediate_hidden_size=64,
                 hidden_size=64, 
                 local_nn_hidden_size=64,
                 local_nn_hidden_layers=2,
                 global_nn_hidden_size=64,
                 global_nn_hidden_layers=2,
                 hidden_layers=4,
                 spatial_dimensions=2
)

model = GraphIterativeWrapper(net)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(1):
    model.train()
    for i, (x, y, p) in enumerate(train_loader):
        x, y, p = x[:,0].float(), y.float(), p.float() # only use the first channel and convert to float32
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

test_loader = DataLoader(advection_test_iterator, batch_size=16, shuffle=False)

model.eval()

loss_values = []
for i, (x, y, p) in enumerate(test_loader):
    x, y, p = x[:,0].float(), y.float(), p.float() # only use the first channel and convert to float32
    y_pred = model(x, p, t_eval=range(1,17))
    loss = criterion(y_pred, y)
    loss_values.append(loss.item())

print(f"Mean Loss: {sum(loss_values) / len(loss_values)}")
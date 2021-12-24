import torch
from dadnet.model.feed_forward import FeedForward

x = torch.randn(18, 32, 32)
y = torch.randint(0, 10, (18,))

model = FeedForward(x.shape, 10)
yhat = model(x)
loss = torch.nn.CrossEntropyLoss()(yhat, y)
loss.backward()
print(loss.item())


x = torch.randn(18, 32, 32)
y = torch.randint(0, 10, (18,))

model = FeedForward(x.shape, 10, hidden_dims=[512, 256, 128, 64, 32, 16])
yhat = model(x)
loss = torch.nn.CrossEntropyLoss()(yhat, y)
loss.backward()
print(loss.item())

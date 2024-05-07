import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('loss')
parser.add_argument('rep', type=int)
parser.add_argument('--datadir', default='/data')
parser.add_argument('--lamda', type=float)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--mlp', action='store_true')
args = parser.parse_args()

from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from time import time
import torch
import losses, data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATASET ##############################

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((int(256*1.05), int(256*1.05)), antialias=True),
    transforms.RandomCrop((256, 256)),
    transforms.ColorJitter(0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ds = getattr(data, args.dataset)
ds = ds(args.datadir, transform, 'train', args.rep)
K = ds.K
tr = DataLoader(ds, args.batchsize, True, num_workers=8, pin_memory=True)

############################## LOSS ##############################

if args.lamda is None:
    loss_fn = getattr(losses, args.loss)(K)
    fname = f'model-{args.dataset}-{args.loss}-{args.rep}.pth'
else:
    loss_fn = getattr(losses, args.loss)(K, args.lamda)
    fname = f'model-{args.dataset}-{args.loss}-{args.rep}-lambda-{args.lamda}.pth'

############################## MODEL ##############################

if args.mlp:
    class MLP(torch.nn.Module):
        def __init__(self, ninputs, hidden, noutputs):
            self.dense1 = torch.nn.Linear(ninputs, 128) 
            self.dense2 = torch.nn.Linear(128, noutputs)
        def forward(self, x):
            x = self.dense1(x)
            x = torch.nn.functional.relu(x)
            x = self.dense2(x)
            return x
    model = MLP()
else:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, loss_fn.how_many_outputs())
model = model.to(device)
loss_fn.to(device)
model.loss_fn = loss_fn

############################## TRAIN OPTS ##############################

opt = torch.optim.Adam(model.parameters(), 1e-4)

############################## TRAIN ##############################

model.train()
for epoch in range(args.epochs):
    print(f'* Epoch {epoch+1}/{args.epochs}')
    tic = time()
    avg_loss = 0
    for X, Y in tr:
        X = X.to(device)
        Y = Y.to(device)
        Yhat = model(X)
        loss_value = loss_fn(Yhat, Y).mean()
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        avg_loss += float(loss_value) / len(tr)
    toc = time()
    print(f'- {toc-tic:.0f}s - Loss: {avg_loss}')

torch.save(model.cpu(), fname)

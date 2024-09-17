# We are following the same recipe as https://arxiv.org/abs/1911.10720

# silence warning when using this without GPU
import warnings
warnings.filterwarnings('ignore', message="Can't initialize NVML")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('loss')
parser.add_argument('--datadir', default='/data/ordinal')
args = parser.parse_args()

from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
from time import time
import torch
import metrics, losses, data
from models import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATASET ##############################

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

############################## LOSS ##############################

ds = getattr(data, args.dataset)
K = ds(args.datadir, transform, 'test', 0).K
loss_fn = getattr(losses, args.loss)(K)

############################## EVAL ##############################

def compute_metrics(rep, metrics_list, lamda):
    ts = ds(args.datadir, transform, 'test', rep)
    ts = DataLoader(ts, 64, True, num_workers=4, pin_memory=True)
    if lamda is None:
        model_name = f'model-{args.dataset}-{args.loss}-{rep}.pth'
    else:
        model_name = f'model-{args.dataset}-{args.loss}-{rep}-lambda-{lamda}.pth'
    model = torch.load(model_name, map_location=device)
    model.eval()
    YY_pred = []
    YY_true = []
    for X, Y in ts:
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            Yhat = model(X)
        YY_pred.append(Yhat)
        YY_true.append(Y)
    PP_pred = loss_fn.to_proba(torch.cat(YY_pred))
    YY_pred = loss_fn.to_classes(PP_pred)
    YY_true = torch.cat(YY_true)
    return torch.tensor([metric(PP_pred, YY_pred, YY_true) for metric in metrics_list], device=device)

metrics_list = [
    metrics.mae,
]

lambdas = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
results = [compute_metrics(0, metrics_list, lamda)[0] for lamda in lambdas]
print(lambdas[torch.argmin(torch.stack(results))])

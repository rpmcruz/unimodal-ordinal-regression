# We are following the same recipe as https://arxiv.org/abs/1911.10720

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('loss')
parser.add_argument('--datadir', default='/data')
parser.add_argument('--lamda', type=float)
parser.add_argument('--rep', type=int, default=1)
args = parser.parse_args()

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import metrics, losses, data
import matplotlib.pyplot as plt

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

def obtain_predictions(rep):
    ts = ds(args.datadir, transform, 'test', rep)
    ts = DataLoader(ts, 64, num_workers=4, pin_memory=True)
    if args.lamda is None:
        model_name = f'model-{args.dataset}-{args.loss}-{rep}.pth'
    else:
        model_name = f'model-{args.dataset}-{args.loss}-{rep}-lambda-{args.lamda}.pth'
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
    YY_true = torch.cat(YY_true)
    return YY_true, PP_pred

import wasserstein
import numpy as np
def is_unimodal(pp):
    xx = np.arange(K)[:, None]
    pp = pp.cpu().detach().numpy()
    #pp = np.around(pp, 3)
    pp /= pp.sum()
    return min(wasserstein.unimodal_wasserstein(xx, pp, mode)[0] for mode in range(K)) < 1e-3

Ytrue, Ppred = obtain_predictions(args.rep)
for ytrue, ppred in zip(Ytrue, Ppred):
    plt.bar(range(K), ppred.cpu())
    plt.xticks(range(K), [f'k={k}' for k in range(K)])
    for i, v in enumerate(ppred):
        plt.text(i, v, f'{v*100:.2f}%', ha='center')
    plt.title(f'true k={int(ytrue)} unimodal={is_unimodal(ppred)}')
    plt.suptitle(f'{args.dataset} {args.loss}')
    plt.show()

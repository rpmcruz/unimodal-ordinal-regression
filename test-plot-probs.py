import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['ICIAR', 'HCI', 'FGNET', 'SMEAR2005', 'FOCUSPATH'])
parser.add_argument('strategy', choices=['allcorrect', 'first', 'quantiles'])
parser.add_argument('n', type=int)
parser.add_argument('--classes', nargs='+', type=int)
parser.add_argument('--interval', default=21, type=int)
parser.add_argument('--split', default='test')
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--models', nargs='+')
args = parser.parse_args()

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import data, losses
import sys
import math, numpy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## OPTIONS ##############################

our_methods = frozenset(('UnimodalNet', 'WassersteinUnimodal_Wass', 'WassersteinUnimodal_KLDIV'))
losses_map = {
    'CrossEntropy': 'CE',
    'POM': 'POM',
    'CDW_CE': 'CDW',
    'OrdinalEncoding': 'OE',
    'BinomialUnimodal_CE': 'BU',
    'PoissonUnimodal': 'PU',
    'CO2': 'CO2',
    'HO2': 'HO2',
    'UnimodalNet': 'UN*',
    'CrossEntropy_UR': 'UR',
    'WassersteinUnimodal_Wass': 'WU-Wass*',
    'WassersteinUnimodal_KLDIV': 'WU-KLDiv*',
}

############################## DATASET ##############################

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tr = getattr(data, args.dataset)('/data/ordinal', transform, args.split, args.fold)
K = tr.K
if args.strategy == 'first':
    rand = numpy.random.default_rng(123)
    tr = [(tr[i], tr.files[i]) for i in rand.choice(len(tr), len(tr), False)]
else:
    tr = [(tr[i], tr.files[i]) for i in range(len(tr))]

data = []
if args.strategy == 'first':
    if args.classes != None:
        # get the first image (from the testing set for fold=0) of each of the
        # requested classes
        for klass in args.classes:
            found = False
            for (x, y), fname in tr:
                if y == klass:
                    found = True
                    data.append((x, y, fname))
                    break
            if not found:
                print('Warning: NOT found klass:', klass, file=sys.stderr)
    else:
        for (x, y), fname in tr:
            data.append((x, y, fname))
            if len(data) >= args.n:
                break
elif args.strategy == 'quantiles':
    n = (K//args.n)+1
    begin = end = 0
    for i in range(args.n):
        begin = end
        end += n
        for (x, y), fname in tr:
            if begin <= y < end:
                data.append((x, y, fname))
                break
else:  # allcorrect
    # find images for which all models are correct
    for (x, y), fname in tqdm(tr):
        x = x.to(device)
        all_correct = True
        for model_i, model in enumerate(args.models):
            loss = model.split('-')[2]
            model = torch.load(model, map_location=device)
            loss_fn = model.loss_fn if hasattr(model, 'loss_fn') else getattr(losses, loss)(K)
            model.eval()
            with torch.no_grad():
                pred = model(x[None])
                prob = loss_fn.to_proba(pred)
                klass = loss_fn.to_classes(prob)
            #print(y, klass[0])
            if klass[0] != y:
                all_correct = False
        if all_correct:
            data.append((x, y, fname))

############################## EVAL ##############################

print(r'\documentclass[preprint,12pt,review]{elsarticle}')
print(r'\usepackage[table]{xcolor}')
print(r'\usepackage{pgfplots}')
print(r'\pgfplotsset{compat=1.18}')
print(r'\begin{document}')
print(r'\begin{figure*}')
print(r'\setlength{\tabcolsep}{0pt}')
print(r'\makebox[\textwidth]{%')
print(r'\begin{tabular}{ccccccc}')
first = True
for image, label, fname in data:
    if not first:
        print(r'\\[-2ex]')
    first = False
    first_class = label - args.interval//2
    second_class = label + args.interval//2
    if first_class < 0:
        second_class += -first_class
        first_class = 0
    if second_class >= K:
        first_class += second_class-K-1
        second_class = K-1
    image = image[None].to(device)
    for model_i, model in enumerate(args.models):
        loss = model.split('-')[2]
        model = torch.load(model, map_location=device)
        loss_fn = model.loss_fn if hasattr(model, 'loss_fn') else getattr(losses, loss)(K)
        model.eval()
        if model_i % 6 == 0:
            if model_i == 0:
                print(f'{{\small$y={label}$}}')
            for _model in args.models[model_i:model_i+6]:
                _loss = _model.split('-')[2]
                clr = r'\cellcolor{black!10}' if _loss in our_methods else ''
                print(r' & {\small' + clr + ' ' + losses_map[_loss] + r'}')
            print(r'\\')
            if model_i == 0:
                print(f'\\raisebox{{9ex}}{{\\multirow{{2}}{{*}}{{{{\includegraphics[width=4em]{{{fname}}}}}}}}}')
        print('&')
        if loss in our_methods:
            print(r'\cellcolor{black!10}')
        with torch.no_grad():
            preds = model(image)
            probs = loss_fn.to_proba(preds)
        print(r'\begin{tikzpicture}[font=\scriptsize]')
        xmax = min(1, torch.ceil(probs[0, first_class:second_class+1].max() / 0.2).item() * 0.2)
        xrange = [f'{x.item():.1f}' for x in torch.arange(0, xmax+0.2, 0.2)]
        if xmax < 0.4:
            xrange = ['0', '0.1', '0.2']
        if xmax == 1:
            xrange = ['0', '0.5', '1']
        if probs[0, first_class:second_class+1].max() < 0.1:
            xmax = probs[0, first_class:second_class+1].max().item()
            decimal_places = -math.floor(math.log10(xmax))+1
            xrange = ('0', numpy.format_float_positional(round(xmax + (0.1**decimal_places), decimal_places)))
            xmax = xrange[-1]
        print(r'\begin{axis}[xbar, width=9.5em, height=18ex, xmin=0, xmax=' + str(xmax) + ',scaled x ticks=false,axis background/.style={fill=white},ytick={' + ','.join(str(k) for k in range(first_class+1, second_class+1, 6)) + '},xticklabels={' + ','.join(xrange) + '},xtick={' + ','.join(xrange) + '}]')
        print(r'\addplot [bar shift=0pt, bar width=0.20ex, fill=gray, draw=gray] coordinates {' + ' '.join(f'({numpy.format_float_positional(probs[0, k])}, {k})' for k in range(first_class, second_class+1) if k != label) + r'};')
        # red for the true label
        print(r'\addplot [bar shift=0pt, bar width=0.20ex, fill=red, draw=red] coordinates {' + f'({numpy.format_float_positional(probs[0, label])},{label})' + r'};')
        print(r'\end{axis}')
        print(r'\end{tikzpicture}')
        if model_i % 6 == 5:
            if model_i == 5:
                print(r'\\[-1ex]')
            else:
                print(r'\\')
    print(r'\hline')
print(r'\end{tabular}')
print('}')
print(f'\caption{{Examples of probabilities outputs for the {args.dataset} dataset. The selection of the examples was made by choosing the first image of each quartile the testing set (fold=0).}}')
print(r'\label{fig:outputs}')
print(r'\end{figure*}')
print(r'\end{document}')

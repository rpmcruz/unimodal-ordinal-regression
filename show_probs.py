import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['ICIAR', 'HCI', 'FGNET', 'SMEAR2005', 'FOCUSPATH'])
parser.add_argument('classes', nargs='+', type=int)
parser.add_argument('--interval', default=21, type=int)
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
    'CDW_CE': 'CE',
    'OrdinalEncoding': 'OE',
    'BinomialUnimodal_CE': 'BU',
    'PoissonUnimodal': 'PU',
    'CO2': 'CO2',
    'HO2': 'HO2',
    'UnimodalNet': 'UN*',
    'WassersteinUnimodal_Wass': 'WU-Wass*',
    'WassersteinUnimodal_KLDIV': 'WU-KLDiv*',
}
losses_list = ['CrossEntropy', 'OrdinalEncoding', 'CDW_CE', 'BinomialUnimodal_CE', 'PoissonUnimodal', 'UnimodalNet', 'WassersteinUnimodal_KLDIV', 'WassersteinUnimodal_Wass', 'CO2', 'HO2']

############################## DATASET ##############################

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tr = getattr(data, args.dataset)('/data/ordinal', transform, 'test', 0)
# get the first image (from the testing set for fold=0) of each of the
# requested classes
images = []
files = []
for klass in args.classes:
    found = False
    for (x, y), fname in zip(tr, tr.files):
        if y == klass:
            found = True
            images.append(x)
            files.append(fname)
            break
    if not found:
        print('Warning: NOT found klass:', klass, file=sys.stderr)

############################## EVAL ##############################

print(r'\documentclass{elsarticle}')
print(r'\usepackage[table]{xcolor}')
print(r'\usepackage{pgfplots}')
print(r'\pgfplotsset{compat=1.18}')
print(r'\usetikzlibrary{patterns}')
print(r'\begin{document}')
print(r'\begin{figure}')
print(r'\setlength{\tabcolsep}{0pt}')
print(r'\makebox[\textwidth]{%')
print(r'\begin{tabular}{cccccc}')
for klass, fname, image in zip(args.classes, files, images):
    first_class = klass - args.interval//2
    second_class = klass + args.interval//2
    image = image[None].to(device)
    for loss_i, loss in enumerate(losses_list):
        if loss_i % 5 == 0:
            if loss_i == 0:
                print(f'{{\small$y={klass}$}}')
            for _loss in losses_list[loss_i:loss_i+5]:
                clr = r'\cellcolor{black!10}' if _loss in our_methods else ''
                print(r' & {\small' + clr + ' ' + losses_map[_loss] + r'}')
            print(r'\\')
            if loss_i == 0:
                print(f'\\raisebox{{1.2ex}}{{\includegraphics[width=5em]{{{fname}}}}}')
        print('&')
        if loss in our_methods:
            print(r'\cellcolor{black!10}')
        model = torch.load(f'model-{args.dataset}-{loss}-0.pth', map_location=device)
        loss_fn = getattr(losses, loss)(tr.K)
        model.eval()
        with torch.no_grad():
            preds = model(image)
            probs = loss_fn.to_proba(preds)
        print(r'\begin{tikzpicture}[font=\footnotesize]')
        xmax = min(1, torch.ceil(probs[:, first_class:second_class+1].max() / 0.2).item() * 0.2)
        xrange = [f'{x.item():.1f}' for x in torch.arange(0, xmax+0.2, 0.2)]
        if xmax < 0.4:
            xrange = ['0', '0.1', '0.2']
        if xmax == 1:
            xrange = ['0', '0.5', '1']
        if probs[:, first_class:second_class+1].max() < 0.1:
            xmax = probs[:, first_class:second_class+1].max().item()
            decimal_places = -math.floor(math.log10(xmax))+1
            xrange = ('0', numpy.format_float_positional(round(xmax + (0.1**decimal_places), decimal_places)))
            xmax = xrange[-1]
        print(r'\begin{axis}[xbar, width=10em, height=22ex, xmin=0, xmax=' + str(xmax) + ',scaled x ticks=false,axis background/.style={fill=white},ytick={' + ','.join(str(k) for k in range(first_class+1, second_class+1, 3)) + '},xticklabels={' + ','.join(xrange) + '},xtick={' + ','.join(xrange) + '}]')
        print(r'\addplot [bar shift=0pt, bar width=0.42ex, pattern=north east lines, pattern color=gray] coordinates {' + ' '.join(f'({numpy.format_float_positional(probs[0, k])}, {k})' for k in range(first_class, second_class+1) if k != klass) + r'};')
        # red for the true label
        print(r'\addplot [bar shift=0pt, bar width=0.42ex, red, pattern=north west lines, pattern color=red] coordinates {' + f'({numpy.format_float_positional(probs[0, klass])},{klass})' + r'};')
        print(r'\end{axis}')
        print(r'\end{tikzpicture}')
        if loss_i % 5 == 4:
            print(r'\\')
    if klass != args.classes[-1]:
        print(r'\hline')
print(r'\end{tabular}')
print('}')
print(f'\caption{{Examples of probabilities outputs for the {args.dataset} dataset. The selection of the examples was made by splitting classes uniformly and choosing the first image from the first fold of the testing set.}}')
print(r'\end{figure}')
print(r'\end{document}')

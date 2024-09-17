# The same datasets as https://arxiv.org/abs/1911.10720
# Breast cancer grading; Photographs dating;  Age estimation

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from skimage.io import imread
import torch
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import StratifiedKFold

REPS = 5

def split(fold, rep, files, labels):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        kfold = StratifiedKFold(REPS, shuffle=True, random_state=123)
        fold = 0 if fold == 'train' else 1
        ix = list(kfold.split(files, labels))[rep][fold]
    return [files[i] for i in ix], [labels[i] for i in ix]

class ImageDataset(Dataset):
    modality = 'image'
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        if fname.lower().endswith('.tif') or fname.lower().endswith('.bmp'):
            img = imread(fname).astype(np.uint8)
            img = np.moveaxis(img, 2, 0)
            img = torch.tensor(img)
        else:
            img = read_image(fname, ImageReadMode.RGB)
        label = self.labels[i]
        if self.transform:
            img = self.transform(img)
        return img, label

class ICIAR(ImageDataset):
    # https://iciar2018-challenge.grand-challenge.org/Dataset/
    K = 4
    def __init__(self, root, transform, fold, rep):
        root = os.path.join(root, 'ICIAR2018_BACH_Challenge', 'Photos')
        classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
        self.files = [os.path.join(root, klass, f) for klass in classes for f in sorted(os.listdir(os.path.join(root, klass))) if f.endswith('.tif')]
        self.labels = [i for i, klass in enumerate(classes) for f in os.listdir(os.path.join(root, klass)) if f.endswith('.tif')]
        self.transform = transform
        self.files, self.labels = split(fold, rep, self.files, self.labels)

class HCI(ImageDataset):
    # http://graphics.cs.cmu.edu/projects/historicalColor/
    def __init__(self, root, transform, fold, rep):
        root = os.path.join(root, 'HistoricalColor-ECCV2012', 'data', 'imgs', 'decade_database')
        decades = sorted(os.listdir(root))
        self.K = len(decades)
        self.files = [os.path.join(root, d, f) for d in decades for f in sorted(os.listdir(os.path.join(root, d))) if not f.startswith('.')]
        self.labels = [i for i, d in enumerate(decades) for f in sorted(os.listdir(os.path.join(root, d))) if not f.startswith('.')]
        self.transform = transform
        self.files, self.labels = split(fold, rep, self.files, self.labels)

class FGNET(ImageDataset):
    # https://yanweifu.github.io/FG_NET_data/
    K = 70
    def __init__(self, root, transform, fold, rep):
        root = os.path.join(root, 'FGNET', 'images')
        files = sorted(os.listdir(root))
        self.files = [os.path.join(root, f) for f in files]
        self.labels = [int(f[4:6]) for f in files]
        self.transform = transform
        self.files, self.labels = split(fold, rep, self.files, self.labels)

class AFAD(ImageDataset):
    # http://afad-dataset.github.io/
    def __init__(self, root, folder, transform, fold, rep):
        root = os.path.join(root, folder)
        ages = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        self.files = [os.path.join(froot, f) for age in ages for froot, _, files in os.walk(os.path.join(root, age)) for f in sorted(files) if f.endswith('.jpg')]
        self.labels = [i for i, age in enumerate(ages) for froot, _, files in os.walk(os.path.join(root, age)) for f in sorted(files) if f.endswith('.jpg')]
        self.transform = transform
        self.files, self.labels = split(fold, rep, self.files, self.labels)

class AFAD_Lite(AFAD):
    K = 39-18+1
    def __init__(self, root, transform, fold, rep):
        super().__init__(root, 'AFAD-Lite', transform, fold, rep)

class AFAD_Full(AFAD):
    K = 75-15+1
    def __init__(self, root, transform, fold, rep):
        super().__init__(root, 'AFAD-Full', transform, fold, rep)

class SMEAR2005(ImageDataset):
    # http://mde-lab.aegean.gr/index.php/downloads
    K = 5
    def __init__(self, root, transform, fold, rep):
        class_folders = {1: 'normal_superficiel', 2: 'normal_intermediate',
            3: 'normal_columnar', 4: 'light_dysplastic', 5: 'moderate_dysplastic'}
        root = os.path.join(root, 'smear2005', 'New database pictures')
        self.files = [os.path.join(root, d, f) for _, d in class_folders.items() for f in sorted(os.listdir(os.path.join(root, d))) if f.endswith('BMP')]
        self.labels = [k-1 for k, d in class_folders.items() for f in sorted(os.listdir(os.path.join(root, d))) if f.endswith('BMP')]
        self.transform = transform
        self.files, self.labels = split(fold, rep, self.files, self.labels)

class FOCUSPATH(ImageDataset):
    # https://zenodo.org/record/3926181
    K = 12
    def __init__(self, root, transform, fold, rep):
        df = pd.read_excel(os.path.join(root, 'focuspath', 'DatabaseInfo.xlsx'))
        root = os.path.join(root, 'focuspath', 'FocusPath_full')
        self.files = [os.path.join(root, row['Name'][:-3] + 'png') for _, row in df.iterrows()]
        self.labels = [min(abs(row['Subjective Score']), 11) for _, row in df.iterrows()]
        self.transform = transform
        self.files, self.labels = split(fold, rep, self.files, self.labels)

class TabularDataset(Dataset):
    modality = 'tabular'
    def __init__(self, root, fname, sep, cols_ignore, cols_category, col_label, labels, discretize_nbins, fold, rep):
        fname = os.path.join(root, 'UCI', fname)
        df = pd.read_csv(fname, header=None, sep=sep)
        X = df.drop(columns=df.columns[col_label])
        Y = df.iloc[:, col_label]
        if cols_ignore:
            X.drop(columns=X.columns[cols_ignore], inplace=True)
        X = pd.get_dummies(X, columns=cols_category).to_numpy(np.float32)
        X = (X-X.mean(0)) / X.std(0)  # z-normalization
        if discretize_nbins:
            Y = Y.to_numpy(np.int64)-1
            bins = np.linspace(0, Y.max(), discretize_nbins, False)
            Y = np.digitize(Y, bins)-1
        elif labels:
            Y = np.array([labels.index(y) for y in Y], np.int64)
        else:
            Y = Y.to_numpy(np.int64)-1
        self.K = Y.max()+1
        self.X, self.Y = split(fold, rep, X, Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def ABALONE5(root, transform, fold, rep):
    return TabularDataset(root, 'abalone.data', ',', None, None, -1, None, 5, fold, rep)
def ABALONE10(root, transform, fold, rep):
    return TabularDataset(root, 'abalone.data', ',', None, None, -1, None, 10, fold, rep)
def BALANCE_SCALE(root, transform, fold, rep):
    return TabularDataset(root, 'balance-scale.data', ',', None, None, -1, None, None, fold, rep)
def CAR(root, transform, fold, rep):
    return TabularDataset(root, 'car.data', ',', None, None, -1, ['unacc', 'acc', 'good', 'vgood'], None, fold, rep)
def LENSES(root, transform, fold, rep):
    return TabularDataset(root, 'lenses.data', r'\s+', [0], [1, 2, 3, 4], -1, None, None, fold, rep)
def NEW_THYROID(root, transform, fold, rep):
    return TabularDataset(root, 'new-thyroid.data', ',', None, None, 0, None, None, fold, rep)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--datadir', default='/data/ordinal')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    ds = globals()[args.dataset]
    tr = ds(args.datadir, None, 'train', 0)
    ts = ds(args.datadir, None, 'test', 0)
    ds = tr
    N = len(tr)+len(ts)
    K = ds.K
    print('N:', N)
    print('K:', K)
    # IR comes from https://ieeexplore.ieee.org/document/6940273
    Nk = np.bincount([y for _, y in tr] + [y for _, y in ts])
    # remove zeros, otherwise we cannot compute this
    Nk = [n for n in Nk if n > 0]
    Nj = [sum(Nk[j] for j in range(len(Nk)) if j != k) for k in range(len(Nk))]
    print('Nk:', Nk)
    print('Nj:', Nj)
    IRk = [Nj[k]/((len(Nk)-1)*Nk[k]) for k in range(len(Nk))]
    print('IRk:', IRk)
    IR = np.mean(IRk)
    print('IR:', IR)
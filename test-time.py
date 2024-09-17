import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--first', nargs='+')
parser.add_argument('--second', nargs='+')
args = parser.parse_args()

from models import MLP
import torch
import numpy as np
times_first = [torch.load(m).train_time for m in args.first]
times_second = [torch.load(m).train_time for m in args.second]
print('Averages:', np.mean(times_first), np.mean(times_second))
print('Medians:', np.median(times_first), np.median(times_second))

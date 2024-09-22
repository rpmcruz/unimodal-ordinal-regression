# make the best values in bold

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
args = parser.parse_args()

import re
which_best = {
    r'\%Accuracy': 100,
    'MAE': 0,
    'QWK': 100,
    r'\%Unimodal': 100,
    'ZME': 0,
    'NLL': 0,
}

f = open(args.results)
for line in f:
    for metric, best_value in which_best.items():
        if line.startswith(metric):
            values = line[:-4].split(' & ')
            avgs = [re.search(r'^\$(-?\d+\.\d+)', v)[1] for v in values[1:]]
            stds = [re.search(r'(-?\d+\.\d+)\$$', v)[1] for v in values[1:]]
            min_dist = min(abs(float(v)-best_value) for v in avgs)
            # 100.0 => 100
            avgs = ['100' if a == '100.0' else a for a in avgs]
            avgs = [r'\mathbf{' + v + '}' if abs(float(v)-best_value) <= min_dist else v for v in avgs]
            print(metric + ' & ' + ' & '.join('$' + avg + r'\color{gray}\pm' + std + '$' for avg, std in zip(avgs, stds)) + r' \\')
            break
    else:
        print(line, end='')
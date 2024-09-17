from scipy.stats import rankdata

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('fname')
parser.add_argument('--datasets', nargs='+')
args = parser.parse_args()

f = open(args.fname)

precision = [1, 1, 2, 1]
averages = {'Acc': [0]*12, 'QWK': [0]*12, 'MAE': [0]*12, 'Uni': [0]*12}
average_ranks = {'Acc': [0]*12, 'QWK': [0]*12, 'MAE': [0]*12, 'Uni': [0]*12}
methods = ['CE', 'POM', 'OE', 'CDW', 'BU', 'PU', 'UN*', 'UR', 'CO2', 'HO2', 'WU-KLDIV*', 'WU-Wass*']
bigger_better = {'Acc': True, 'QWK': True, 'MAE': False, 'Uni': True}

N = 10 if args.datasets == None else len(args.datasets)
skip_dataset = False

for line in f:
    fields = line.split('&')
    if len(fields) < 5:
        continue
    if args.datasets != None:
        if not any(avg in fields[0] for avg in averages):
            skip_dataset = not any(dataset in fields[0] for dataset in args.datasets)
    if skip_dataset:
        continue
    for avg in averages:
        if avg in fields[0]:
            results = [float(f[2:].split(r'\color')[0]) for f in fields[1:]]
            if bigger_better[avg]:
                ranks = 13 - rankdata(results, method='max')
            else:
                ranks = rankdata(results, method='min')
            for method in range(12):
                averages[avg][method] += results[method] / N
                average_ranks[avg][method] += ranks[method] / N

f.close()

print(r'\documentclass{standalone}')
print(r'\begin{document}')
print(r'\begin{tabular}{lllllllll}')
for method in range(12):
    print(methods[method], end=' & ')
    for i, avg in enumerate(averages):
        print(f"{averages[avg][method]:.{precision[i]}f}", end=' & ')
        print(f"{average_ranks[avg][method]:.1f}", end='')
        print(r' \\' if i == len(averages)-1 else ' & ', end='')
    print()
print(r'\end{tabular}')
print(r'\end{document}')

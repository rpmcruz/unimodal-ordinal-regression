from scipy.stats import rankdata

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('fname')
parser.add_argument('--datasets', nargs='+')
args = parser.parse_args()

f = open(args.fname)

methods = ['CE', 'POM', 'OE', 'CDW', 'BU', 'PU', 'ORD-ACL', 'VS-SL', 'UN*', 'UR', 'CO2', 'WU-KLDIV*', 'WU-Wass*']
precision = [1, 1, 1, 2, 1, 2, 2]
averages = {'Acc': [0]*len(methods), 'QWK': [0]*len(methods), r'$\tau$': [0]*len(methods), 'MAE': [0]*len(methods), 'Uni': [0]*len(methods), 'ZME': [0]*len(methods), 'NLL': [0]*len(methods)}
average_ranks = {'Acc': [0]*len(methods), 'QWK': [0]*len(methods), r'$\tau$': [0]*len(methods), 'MAE': [0]*len(methods), 'Uni': [0]*len(methods), 'ZME': [0]*len(methods), 'NLL': [0]*len(methods)}
bigger_better = {'Acc': True, 'QWK': True, r'$\tau$': True, 'MAE': False, 'Uni': True, 'ZME': False, 'NLL': False}

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
            results = [f[2:].split(r'\color')[0] for f in fields[1:]]
            results = [float(r[8:-1]) if r.startswith(r'\mathbf{') else float(r) for r in results]
            if avg == 'ZME':
                ranks = rankdata([abs(r) for r in results], method='min')
            elif bigger_better[avg]:
                ranks = (len(methods)+1) - rankdata(results, method='max')
            else:
                ranks = rankdata(results, method='min')
            for method in range(len(methods)):
                averages[avg][method] += results[method] / N
                average_ranks[avg][method] += ranks[method] / N

f.close()

print(r'\documentclass{standalone}')
print(r'\begin{document}')
print(r'\begin{tabular}{lllllllll}')
for method in range(len(methods)):
    print(methods[method], end=' & ')
    for i, avg in enumerate(averages):
        print(f"{averages[avg][method]:.{precision[i]}f}", end='')
        #print(f"{average_ranks[avg][method]:.1f}", end='')
        print(r' \\' if i == len(averages)-1 else ' & ', end='')
    print()
print(r'\end{tabular}')
print(r'\end{document}')

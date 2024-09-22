import torch

def acc(ppred, ypred, ytrue):
    return (ypred == ytrue).float().mean()

def mae(ppred, ypred, ytrue):
    return (ypred - ytrue).abs().float().mean()

def f1(ppred, ypred, ytrue):
    tp = torch.logical_and(ytrue, ypred).float().mean()
    fp = torch.logical_and(ytrue == 0, ypred).float().mean()
    fn = torch.logical_and(ytrue, ypred == 0).float().mean()
    return tp / (tp + 0.5*(fp+fn))

def Balanced(fn):
    def f(ppred, ypred, ytrue):
        KK = torch.unique(ytrue)
        return sum(fn(ppred[ytrue == k], ypred[ytrue == k], ytrue[ytrue == k]) for k in KK) / len(KK)
    f.__name__ = f'balanced_{fn.__name__}'
    return f

def MacroAvg(fn):
    def f(ppred, ypred, ytrue):
        KK = torch.unique(ytrue)
        return sum(fn(ppred, (ypred == k).long(), (ytrue == k).long()) for k in KK) / len(KK)
    f.__name__ = f'macro_{fn.__name__}'
    return f

def Percentage(fn):
    def f(ppred, ypred, ytrue):
        return 100*fn(ppred, ypred, ytrue)
    return f

def is_unimodal(p):
    zero = torch.zeros(1, device=p.device)
    p = torch.sign(torch.round(torch.diff(p, prepend=zero, append=zero), decimals=2))
    p = torch.diff(p[p != 0])
    p = p[p != 0]
    return len(p) <= 1

def times_unimodal_wasserstein(ppred, ypred, ytrue):
    return sum(is_unimodal(p) for p in ppred) / len(ppred)

def confusion_matrix(ypred, ytrue):
    num_classes = 1+max(ypred.amax(), ytrue.amax())
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(ytrue, ypred):
        confusion[t, p] += 1
    return confusion

def quadratic_weighted_kappa(ppred, ypred, ytrue):
    O = confusion_matrix(ytrue, ypred)  # observed agreement
    E = torch.outer(O.sum(0), O.sum(1))  # expected agreement
    # normalize
    O = O / torch.sum(O)
    E = E / torch.sum(E)
    # quadratic weighted kappa
    n = O.shape[0]
    weights = ((torch.arange(n) - torch.arange(n)[:, None])**2) / ((n-1)**2)
    num = torch.sum(weights * O)
    den = torch.sum(weights * E)
    return 1 - (num / den)

def zero_mean_error(ppred, ypred, ytrue):
    # https://www.youtube.com/watch?v=uRb6Fi44ww0
    # the average of the residuals should be zero
    residuals = ypred - ytrue
    return torch.mean(residuals.float())

def negative_log_likelihood(ppred, ypred, ytrue):
    # ce = -(y=1)*log(p̂) <=> -log(p̂[y])
    ppred = ppred[range(len(ytrue)), ytrue]
    ppred = ppred[ppred > 1e-8]  # avoid log(0)
    return torch.mean(-torch.log(ppred))

def spearman(ppred, ypred, ytrue):
    import scipy.stats
    return scipy.stats.spearmanr(ypred.cpu(), ytrue.cpu()).statistic

def kendall_tau(ppred, ypred, ytrue):
    import scipy.stats
    return scipy.stats.kendalltau(ypred.cpu(), ytrue.cpu()).statistic

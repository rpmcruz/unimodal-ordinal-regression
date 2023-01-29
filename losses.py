import torch
import torch.nn.functional as F

############################### UTILITIES ######################################

def fact(x):
    return torch.exp(torch.lgamma(x+1))

def log_fact(x):
    return torch.lgamma(x+1)

def to_classes(probs, method=None):
    # method can be:
    # "mode" = class with highest probability (argmax) [default]
    # "mean" = expectation average of the probabilities distribution
    # "median" = median weighted by the probabilities distribution
    assert method in (None, 'mode', 'mean', 'median')
    if method == 'mean':  # also called expectation trick by Beckham
        K = probs.shape[1]
        kk = torch.arange(K, device=probs.device, dtype=torch.float32)[None]
        return torch.round(torch.sum(kk * probs, 1)).long()
    elif method == 'median':
        # the weighted median is the value whose cumulative probability is 0.5
        Pc = torch.cumsum(probs, 1)
        return torch.sum(Pc < 0.5, 1)
    else:  # default=mode
        return probs.argmax(1)

# we are using softplus instead of relu since it is smoother to optimize.
# as in http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
approx_relu = F.softplus
ce = torch.nn.CrossEntropyLoss(reduction='none')

################################# LOSSES #######################################

class OrdinalLoss(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def how_many_outputs(self):
        # how many output neurons does this loss require?
        return self.K

    def forward(self, ypred, ytrue):
        # computes the loss
        pass

    def reset_epoch(self):
        # some losses use this method to reset running statistics used, e.g., to
        # compute thresholds
        pass

    def to_proba(self, ypred):
        # output -> probabilities
        pass

    def to_classes(self, ypred, method=None):
        # output -> classes. for an explanation of the 'method' parameter, see
        # the utility function to_classes() above.
        # note: only in rare cases, you need to overload this (e.g., if your
        # loss does not produce probabilities or if it has a special means of
        # computing classes). otherwise, do not overload.
        probs = self.to_proba(ypred)
        classes = to_classes(probs, method)
        return classes

################################################################################
# Classical losses.                                                            #
################################################################################

class CrossEntropy(OrdinalLoss):
    def forward(self, ypred, ytrue):
        return ce(ypred, ytrue)

    def to_proba(self, ypred):
        return F.softmax(ypred, 1)

class MAE(OrdinalLoss):
    def how_many_outputs(self):
        return 1

    def forward(self, ypred, ytrue):
        ypred = torch.clamp(ypred, 0, self.K-1)[:, 0]
        return torch.abs(ypred-ytrue)

    def to_proba(self, ypred):
        ypred = torch.clamp(torch.round(ypred), 0, self.K-1)[:, 0].long()
        return F.one_hot(ypred, self.K)

class MSE(MAE):
    def forward(self, ypred, ytrue):
        ypred = torch.clamp(ypred, 0, self.K-1)[:, 0]
        return (ypred-ytrue)**2

class DummyMedian(OrdinalLoss):
    # This is just a useful hack to easily get a baseline benchmark. Always
    # returns the average class.
    def forward(self, ypred, ytrue):
        return 0*ypred

    def to_proba(self, ypred):
        probs = torch.zeros(len(ypred), self.K, device=ypred.device)
        probs[:, self.K//2] = 1
        return probs

################################################################################
# Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network        #
# approach to ordinal regression." 2008 IEEE international joint conference on #
# neural networks (IEEE world congress on computational intelligence). IEEE,   #
# 2008. https://arxiv.org/pdf/0704.1028.pdf                              #
################################################################################
# Notice that some authors cite later papers like OR-CNN (Zhenxing Niu et al,  #
# 2016) but we believe this was the first for neural networks and is based on  #
# the Frank & Hall (2001) ordinal ensemble.                                    #
################################################################################

class OrdinalEncoding(OrdinalLoss):
    def how_many_outputs(self):
        return self.K-1

    def forward(self, ypred, ytrue):
        # if K=4, then
        #                k = 0  1  2
        #     Y=0 => P(Y>k)=[0, 0, 0]
        #     Y=1 => P(Y>k)=[1, 0, 0]
        #     Y=2 => P(Y>k)=[1, 1, 0]
        #     Y=3 => P(Y>k)=[1, 1, 1]
        KK = torch.arange(self.K-1, device=ytrue.device).expand(len(ytrue), -1)
        yytrue = (ytrue[:, None] > KK).float()
        return torch.sum(F.binary_cross_entropy_with_logits(ypred, yytrue, reduction='none'), 1)

    def to_proba(self, ypred):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y>k) into P(Y=k)
        # P(Y=0) = 1-P(Y>0)
        # P(Y=1) = P(Y>0)-P(Y>1)
        # ...
        # P(Y=K-1) = P(Y>K-2)
        probs = torch.sigmoid(ypred)
        prob_0 = 1-probs[:, [0]]
        prob_k = probs[:, [-1]]
        probs = torch.cat((prob0, probs[:, :-1]-probs[:, 1:], prob_k), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return probs

    def to_classes(self, ypred, method=None):
        # for OrdinalEncoding, if method=None (default) use the cumulative
        # distribution directly to get the classes, as suggested in the paper.
        if method is None:
            # notice we are working on the logit space, therefore yp>0 is the
            # same as sigmoid(yp)>0.5
            return torch.sum(ypred >= 0, 1)
        return super().to_classes(ypred, method)

################################################################################
# Polat, Gorkem, et al. "Class Distance Weighted Cross-Entropy Loss for        #
# Ulcerative Colitis Severity Estimation." arXiv preprint arXiv:2202.05167     #
# (2022). https://arxiv.org/pdf/2202.05167.pdf                                 #
################################################################################
# Castagnos, François, Martin Mihelich, and Charles Dognin. "A Simple Log-     #
# -based Loss Function for Ordinal Text Classification." Proceedings of the    #
# 29th International Conference on Computational Linguistics. 2022.            #
# https://aclanthology.org/2022.coling-1.407.pdf                               #
################################################################################
# These two papers propose something identical. Not sure which paper came      #
# first. Interestingly, CDW_CE recommends alpha=5, while OrdinalLogLoss        #
# recommends alpha=1.5 (which for us also works better).                       #
################################################################################

class CDW_CE(OrdinalLoss):
    def __init__(self, K, alpha=5):
        super().__init__(K)
        self.alpha = alpha

    def d(self, y):
        # internal function for the distance penalization. you may overload
        # this function if you want to use another.
        i = torch.arange(self.K, device=y.device)
        return torch.abs(i[None] - y[:, None])**self.alpha

    def forward(self, ypred, ytrue):
        ypred = F.softmax(ypred, 1)
        return -torch.sum(torch.log(1-ypred) * self.d(ytrue), 1)

    def to_proba(self, ypred):
        return F.softmax(ypred, 1)

################################################################################
# da Costa, Joaquim F. Pinto, Hugo Alonso, and Jaime S. Cardoso. "The unimodal #
# model for the classification of ordinal data." Neural Networks 21.1 (2008):  #
# 78-91.                                                                       #
# https://www.sciencedirect.com/science/article/pii/S089360800700202X          #
################################################################################

class BinomialUnimodal_CE(OrdinalLoss):
    def how_many_outputs(self):
        return 1

    def forward(self, ypred, ytrue):
        return F.nll_loss(self.to_log_proba(ypred), ytrue, reduction='none')

    def to_proba(self, ypred):
        device = ypred.device
        probs = torch.sigmoid(ypred)
        N = ypred.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = fact(K-1) * (probs**kk) * (1-probs)**(K-kk-1)
        den = fact(kk) * fact(K-kk-1)
        return num / den

    def to_log_proba(self, ypred):
        # used internally by the loss in forward()
        device = ypred.device
        log_probs = F.logsigmoid(ypred)
        log_inv_probs = F.logsigmoid(-ypred)
        N = ypred.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = log_fact(K-1) + kk*log_probs + (K-kk-1)*log_inv_probs
        den = log_fact(kk) + log_fact(K-kk-1)
        return num - den

class BinomialUnimodal_MSE(BinomialUnimodal_CE):
    def forward(self, ypred, ytrue):
        device = ypred.device
        probs = self.to_proba(ypred)
        yonehot = torch.zeros(probs.shape[0], self.K, device=device)
        yonehot[range(probs.shape[0]), ytrue] = 1
        return torch.sum((probs - yonehot)**2, 1)

################################################################################
# Beckham, Christopher, and Christopher Pal. "Unimodal probability             #
# distributions for deep ordinal classification." International Conference on  #
# Machine Learning. PMLR, 2017.                                                #
# http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf                   #
################################################################################

class PoissonUnimodal(OrdinalLoss):
    def how_many_outputs(self):
        return 1

    def forward(self, ypred, ytrue):
        return ce(self.activation(ypred), ytrue)

    def to_proba(self, ypred):
        return F.softmax(self.activation(ypred), 1)

    def activation(self, ypred):
        # internal function used by forward() and to_proba()
        # they apply softplus (relu) to avoid log(negative)
        ypred = F.softplus(ypred)
        KK = torch.arange(1., self.K+1, device=ypred.device)[None]
        return KK*torch.log(ypred) - ypred - log_fact(KK)

################################################################################
# Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Ordinal losses for   #
# classification of cervical cancer risk." PeerJ Computer Science 7 (2021):    #
# e457. https://peerj.com/articles/cs-457/                                     #
################################################################################
# These losses require two parameters: omega and lambda.                       #
# The default omega value comes from the paper.                                #
# The default lambda values comes from our experiments.                        #
################################################################################

def entropy_term(ypred):
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    P = F.softmax(ypred, 1)
    logP = F.log_softmax(ypred, 1)
    return -torch.sum(P * logP, 1)

def neighbor_term(ypred, ytrue, margin):
    margin = torch.tensor(margin, device=ytrue.device)
    P = F.softmax(ypred, 1)
    K = P.shape[1]
    dP = torch.diff(P, 1)
    sign = (torch.arange(K-1, device=ytrue.device)[None] >= ytrue[:, None])*2-1
    return torch.sum(approx_relu(margin + sign*dP, 1))

class CO2(OrdinalLoss):
    def __init__(self, K, lamda=0.01, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def forward(self, ypred, ytrue):
        term = neighbor_term(ypred, ytrue, self.omega)
        return ce(ypred, ytrue) + self.lamda*term

    def to_proba(self, ypred):
        return F.softmax(ypred, 1)

class CO(CO2):
    # CO is the same as CO2 with omega=0
    def __init__(self, K, lamda=0.01):
        super().__init__(K, lamda, 0)

class HO2(OrdinalLoss):
    def __init__(self, K, lamda=1.0, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def forward(self, ypred, ytrue, reduction='mean'):
        term = neighbor_term(ypred, ytrue, self.omega)
        return entropy_term(ypred) + self.lamda*term

    def to_proba(self, ypred):
        return F.softmax(ypred, 1)


################################################################################
# PROPOSED LOSSES.                                                             #
# Jaime S. Cardoso, Ricardo Cruz and Tomé Albuquerque, 2022.                   #
################################################################################

class UnimodalNet(OrdinalLoss):
    def forward(self, ypred, ytrue):
        return ce(self.activation(ypred), ytrue)

    def to_proba(self, ypred):
        return F.softmax(self.activation(ypred), 1)

    def activation(self, ypred):
        # first use relu: we need everything positive
        # for differentiable reasons, we use leaky relu
        ypred = approx_relu(ypred)
        # if output=[X,Y,Z] => pos_slope=[X,X+Y,X+Y+Z]
        # if output=[X,Y,Z] => neg_slope=[Z,Z+Y,Z+Y+X]
        pos_slope = torch.cumsum(ypred, 1)
        neg_slope = torch.flip(torch.cumsum(torch.flip(ypred, [1]), 1), [1])
        ypred = torch.minimum(pos_slope, neg_slope)
        return ypred

def unimodal_wasserstein(p, mode):
    # Returns the closest unimodal distribution to p with the given mode.
    # Return tuple:
    # 0: total transport cost
    # 1: closest unimodal distribution
    import numpy as np
    from scipy.spatial.distance import squareform, pdist
    from scipy.optimize import linprog
    assert abs(p.sum()-1) < 1e-6, 'Expected normalized probability mass.'
    assert np.any(p >= 0), 'Expected nonnegative probabilities.'
    assert len(p.shape) == 1, 'Probabilities p must be a vector.'
    assert 0 <= mode < p.size, 'Invalid mode value.'
    K = p.size
    C = squareform(pdist(np.arange(K)[:, None]))  # cost matrix
    Ap = [([0]*i + [1] + [0]*(K-i-1))*K for i in range(K)]
    Ai = [[0]*i*K + [1]*K + [-1]*K + [0]*(K-i-2)*K if i < mode else
          [0]*i*K + [-1]*K + [1]*K + [0]*(K-i-2)*K for i in range(K-1)]
    result = linprog(C.ravel(), A_ub=Ai, b_ub=np.zeros(K-1), A_eq=Ap, b_eq=p)
    T = result.x.reshape(K, K)
    return (T*C).sum(), T.sum(1)

def emd(p, q):
    # https://en.wikipedia.org/wiki/Earth_mover%27s_distance
    pp = p.cumsum(1)
    qq = q.cumsum(1)
    return torch.sum(torch.abs(pp-qq), 1)

def is_unimodal(p):
    # checks (true/false) whether the given probability vector is unimodal. this
    # function is not used by the following classes, but it is used in the paper
    # to compute the "% times unimodal" metric
    zero = torch.zeros(1, device=p.device)
    p = torch.sign(torch.round(torch.diff(p, prepend=zero, append=zero), decimals=2))
    p = torch.diff(p[p != 0])
    p = p[p != 0]
    return len(p) <= 1

class WassersteinUnimodal_KLDIV(OrdinalLoss):
    def __init__(self, K, lamda=100.):
        super().__init__(K)
        self.lamda = lamda

    def forward(self, ypred, ytrue):
        probs = F.softmax(ypred, 1)
        probs_log = F.log_softmax(ypred, 1)
        closest_unimodal = torch.stack([
            torch.tensor(unimodal_wasserstein(phat, y)[1], dtype=torch.float32, device=ytrue.device)
            for phat, y in zip(probs.cpu().detach().numpy(), ytrue.cpu().numpy())])
        term = self.distance_loss(probs, probs_log, closest_unimodal)
        return ce(ypred, ytrue) + self.lamda*term

    def distance_loss(self, phat, phat_log, target):
        return torch.sum(F.kl_div(phat_log, target, reduction='none'), 1)

    def to_proba(self, ypred):
        return F.softmax(ypred, 1)

class WassersteinUnimodal_Wass(WassersteinUnimodal_KLDIV):
    def distance_loss(self, phat, phat_log, target):
        return emd(phat, target)

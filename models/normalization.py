import torch
import torch.nn as nn

from torch.nn import Parameter

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        c=int(C/G)*G
        if c==0:
            c=C
            G=C

        x1=x[:,:c].view(N,G,-1)
        mean = x1.mean(-1, keepdim=True)
        var = x1.var(-1, keepdim=True)

        x1 = (x1 - mean) / (var + self.eps).sqrt()
        x1 = x1.view(N, c, H, W)

        if c!=C:
            x_tmp=x[:, c:].view(N,-1)
            mean=x_tmp.mean(-1).view(N,1,1,1)
            var=x_tmp.var(-1).view(N,1,1,1)
            x_tmp = (x[:, c:] - mean) / (var + self.eps).sqrt()
            x1=torch.cat([x1,x_tmp],dim=1)
        return x1 * self.weight + self.bias


class FeatureDecorr(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5,n_iter=10):
        super(FeatureDecorr, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps
        self.n_iter=n_iter

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups

        c=int(C/G)*G
        if c==0:
            c=C
            G=C

        #x1=x[:,:c].view(N,G,-1)
        x1 = x[:,:c].view(N, int(c/G), G, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, G, -1)

        mean = x1.mean(-1, keepdim=True)
        x_centerred=x1-mean
        cov=torch.bmm(x_centerred,x_centerred.permute(0,2,1))/x_centerred.shape[2]+ self.eps*torch.eye(G,dtype=x.dtype,device=x.device).unsqueeze(0)

        decorr=isqrt_newton_schulz_autograd(cov, self.n_iter)
        x1 = torch.bmm(decorr,x_centerred)
        #x1 = x1.view(N, c, H, W)
        x1 = x1.view(N, G, int(c / G), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N,c,H,W)

        if c!=C:
            x_tmp=x[:, c:].view(N,-1)
            mean=x_tmp.mean(-1).view(N,1,1,1)
            var=x_tmp.var(-1).view(N,1,1,1)
            x_tmp = (x[:, c:] - mean) / (var + self.eps).sqrt()
            x1=torch.cat([x1,x_tmp],dim=1)

        return x1 * self.weight + self.bias


def isqrt_newton_schulz_autograd(A, numIters):
    batchSize,dim,_ = A.shape
    normA=A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)

    return A_isqrt
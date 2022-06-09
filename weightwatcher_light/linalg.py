import torch
import numpy as np
# import cupy as cp
# from cupyx.scipy.sparse.linalg import eigsh, lobpcg, LinearOperator
from math import acos,sin,pi,cos

def own_eigsh(linop, N, k):
    from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
    pos_q_mat, pos_t_mat = lanczos_tridiag(
        linop,
        max(20, 2 * k),
        device="cuda",
        dtype=torch.float32,
        matrix_shape=(N, N),
    )
    # convert the tridiagonal t matrix to the eigenvalues
    pos_eigvals, pos_eigvecs = lanczos_tridiag_to_diag(pos_t_mat)
    # print(pos_eigvals)
    # eigenvalues may not be sorted
    maxeig = torch.max(pos_eigvals)
    return torch.sort(pos_eigvals).values[-k:]


class AlphaEstimator():
    def __init__(self, layer, device, shape_in, shape_out, x_min=None, x_max=None, deg=128):
        self.layer = layer
        self.device = device
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.m = np.prod(shape_out)
        self.n = np.prod(shape_in)
        self.x_max = x_max
        if x_min is None:
            self.coeffs = torch.zeros(4096, deg).to(device)
            self.x_min = np.logspace(-8, np.log10(x_max), self.coeffs.shape[0] // 2)
            # self.x_min = np.linspace(1e-8, x_max, self.coeffs.shape[0] // 2)
        else:
            self.coeffs = torch.zeros(2, deg).to(device)
            self.x_min = np.array([x_min])
        self.deg = deg
        def func(x, xmin):
            return (((x + 1) * (0.5 * x_max)) > x_min) * np.nan_to_num((np.log((x + 1) * (0.5 * x_max) / x_min)))
        def func2(x, xmin):
            return 1.0 * (((x + 1) * (0.5 * x_max)) >= x_min)
        for i, x_min in enumerate(self.x_min):
            self.coeffs[2 * i] = torch.tensor(np.polynomial.chebyshev.chebinterpolate(func, deg - 1, args=(x_min,))).to(device)

            # self.coeffs[2 * i + 1] = torch.tensor(np.polynomial.chebyshev.chebinterpolate(func2, deg - 1, args=(x_min,))).to(device)
        # Closed form Solution for Step Function
        alphaP  =  np.pi / (self.deg + 2)
        a = np.minimum(self.x_min / (0.5 * x_max) - 1, 1)
        b = 1
        self.coeffs[1::2, 0] = torch.tensor(1.0 / pi * (np.arccos(a) - np.arccos(b)))# * \
        # self.coeffs[1::2, 0] *= torch.tensor((sin(1) * alphaP / ((self.deg + 2)*sin(alphaP)) + (1-(1)/(self.deg + 2)) * cos(0 * alphaP)))
        for j in range(1, deg):
            self.coeffs[1::2, j] = torch.tensor(2.0 / pi * (np.sin(j * np.arccos(a)) - np.sin(j * np.arccos(b))) / j)
            # self.coeffs[1::2, j] *= torch.tensor((np.sin(j + 1)* alphaP / ((self.deg + 2) * np.sin(alphaP)) + (1 - (j + 1) / (self.deg + 2)) * np.cos(j * alphaP)))
            # tmp = np.linspace(-1,1,10000)
            # print(np.sqrt(np.mean((np.polynomial.chebyshev.chebval(tmp, self.coeffs[2 * i + 1].cpu().numpy()) - func2(tmp, x_min))**2)))
        self.samples = 1024
        self.X = None

    def matMat(self, W):
        # y = torch.as_tensor(w, device=self.device).float().view(self.shape_out)
        y = W.view(-1, *self.shape_out[1:])
        if isinstance(self.layer, torch.nn.Conv2d):
            yy = torch.nn.functional.conv_transpose2d(y, self.layer.weight, None, self.layer.stride, self.layer.padding, 0, self.layer.groups, self.layer.dilation)
            y = torch.nn.functional.conv2d(yy, self.layer.weight, None, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)
        elif isinstance(self.layer, torch.nn.Linear):
            yy = torch.nn.functional.linear(y, self.layer.weight.T, None)
            y = torch.nn.functional.linear(yy, self.layer.weight, None)
        return (2.0 / self.x_max) * y.view(-1, self.m).detach() - W #  * x - y
    def matMatT(self, W):
        # y = torch.as_tensor(w, device=self.device).float().view(self.shape_out)
        yy = W.view(-1, *self.shape_in[1:])
        if isinstance(self.layer, torch.nn.Conv2d):
            y = torch.nn.functional.conv2d(yy, self.layer.weight, None, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)
            tmp = (torch.tensor(self.shape_in[2:3]) - 1) * torch.tensor(self.layer.stride) - 2 * torch.tensor(self.layer.padding) + torch.tensor(self.layer.dilation) * (torch.tensor(self.layer.weight.shape[2:3]) -1 ) + 1
            output_padding = torch.tensor(self.shape_in[2:3]) - tmp
            yy = torch.nn.functional.conv_transpose2d(y, self.layer.weight, None, self.layer.stride, self.layer.padding, tuple(output_padding), self.layer.groups, self.layer.dilation)
        elif isinstance(self.layer, torch.nn.Linear):
            y = torch.nn.functional.linear(yy, self.layer.weight, None)
            yy = torch.nn.functional.linear(y, self.layer.weight.T, None)
        return (2.0 / self.x_max) * yy.view(-1, self.n).detach() - W 
    def normalSample(self):
        self.X = torch.nn.init.normal_(self.X).sign()

    @torch.no_grad()
    def produceSample(self):
        m = min(self.m, self.n)
        if m == self.m:
            matMat = self.matMat
        else:
            matMat = self.matMatT
        self.X = torch.zeros((self.samples, m)).to(self.device).detach()
        self.normalSample()
        # M2 = torch.zeros((self.samples, self.m)).to(self.device).detach()
        # M2 = M2.new_tensor(self.X).to(self.device).detach()
        M2 = self.X.clone().detach()
        M1 = matMat(M2)

        f = (self.coeffs[:, 0] * torch.bmm(self.X.view(self.samples, 1, m), M2.view(self.samples, m, 1)) +
             self.coeffs[:, 1] * torch.bmm(self.X.view(self.samples, 1, m), M1.view(self.samples, m, 1)))

        for i in range(2, self.deg):
            F = 2.0 * matMat(M1) - M2

            del M2
            M2 = M1
            M1 = F

            f  = f + self.coeffs[:, i] * torch.bmm(self.X.view(self.samples, 1, m), F.view(self.samples, m, 1))

        return f

    def estimateAlpha(self, draws=1):
        t = torch.zeros(draws,self.coeffs.shape[0]).to(self.device)
        for repeats in range(draws):
            particles = self.produceSample()
            t[repeats,:] = particles.mean(dim=(0, 1))
        t, _ = torch.median(t, dim=0) # median of means...for robustness...cause sometimes things go belly-up
        return 1 + t[1::2] / t[::2], t[0::2], t[1::2]
    def fit(self, draws=1):
        alphas, summe, n = self.estimateAlpha(draws=draws)
        # print(n)

        x_min = torch.tensor(self.x_min).to(self.device)
        D_best = 1
        alpha_best = -1
        xmin_best = 0
        n = n.clamp(0, min(self.n, self.m))
        ignore = len(alphas) // 200 + 1 #TODO: This feels hacky...
        for i, (xm, alpha) in enumerate(zip(self.x_min[:-ignore], alphas[:-ignore])):
            empirical_cumulative = (n[i] - n) / n[i]
            model = 1 - (xm / x_min) ** (alpha - 1) #evaluate at the xmin points
            D = (model[i:] - empirical_cumulative[i:]).abs().max()
            # print((model[i:] - empirical_cumulative[i:]).abs().cpu().numpy(), D.item())
            # print(alpha, xm, D, ((model[i:] - empirical_cumulative[i:])**2).mean())
            if D < D_best:
                xmin_best = xm
                alpha_best = alpha
                D_best = D
                which = i

        self.xmin = xmin_best
        self.D = D_best
        
        self.summe = summe[which]
        self.est_n = n[which]
        return alpha_best

class TopEigValsEstimator():
    def __init__(self,layer, device,shape_in,shape_out):
        self.layer = layer
        self.device = device
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.m = np.prod(shape_out)
        self.n = np.prod(shape_in)

    def matVec(self, w):
        y = w.view(self.shape_out)
        if isinstance(self.layer, torch.nn.Conv2d):
            yy = torch.nn.functional.conv_transpose2d(y, self.layer.weight, None, self.layer.stride, self.layer.padding, 0, self.layer.groups, self.layer.dilation)
            y = torch.nn.functional.conv2d(yy, self.layer.weight, None, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)
        elif isinstance(self.layer, torch.nn.Linear):
            yy = torch.nn.functional.linear(y, self.layer.weight.T, None)
            y = torch.nn.functional.linear(yy, self.layer.weight, None)
        return y.view(self.m, 1).detach()
    def matVecT(self,y):
        yy = y.view(self.shape_in)
        if isinstance(self.layer, torch.nn.Conv2d):
            y = torch.nn.functional.conv2d(yy, self.layer.weight, None, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)
            tmp = (torch.tensor(self.shape_in[2:3]) - 1) * torch.tensor(self.layer.stride) - 2 * torch.tensor(self.layer.padding) + torch.tensor(self.layer.dilation) * (torch.tensor(self.layer.weight.shape[2:3]) -1 ) + 1
            output_padding = torch.tensor(self.shape_in[2:3]) - tmp
            yy = torch.nn.functional.conv_transpose2d(y, self.layer.weight, None, self.layer.stride, self.layer.padding, tuple(output_padding), self.layer.groups, self.layer.dilation)
        elif isinstance(self.layer, torch.nn.Linear):
            y = torch.nn.functional.linear(yy, self.layer.weight, None)
            yy = torch.nn.functional.linear(y, self.layer.weight.T, None)
        x = yy.view(self.n, 1).detach()#
        return x

    def estimateEigVals(self,budget):
        if self.m < self.n:
            s = own_eigsh(self.matVec, self.m, k=min(self.m - 1, budget))
        else:
            s = own_eigsh(self.matVecT, self.n, k=min(self.n - 1, budget))
        return s.cpu().numpy()



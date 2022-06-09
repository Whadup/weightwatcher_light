import torch
import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import eigsh, lobpcg, LinearOperator


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


class AlohaEstimator():
    def __init__(self, layer, device, shape_in, shape_out, x_min=None, x_max=None, deg=32):
        
        
        def log(x):
            #log((x*15 + 15) / 0.1), log(30/0.1)
            return np.log((x + 1) * (0.5 * x_max))
        self.layer = layer
        self.device = device
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.m = np.prod(shape_out)
        self.n = np.prod(shape_in)
        self.x_max = x_max
        if x_min is None:
            self.coeffs = torch.zeros(128, deg).to(device)
            self.x_min = np.logspace(-16, 0, self.coeffs.shape[0])
        else:
            self.coeffs = torch.zeros(1, deg).to(device)
            self.x_min = np.array([x_min])
        for i, x_min in enumerate(self.x_min):
            def func(x):
                return (((x + 1) * (0.5 * x_max)) > x_min) * np.nan_to_num((np.log((x + 1) * (0.5 * x_max) / x_min)))
            self.coeffs[i] = torch.tensor(np.polynomial.chebyshev.chebinterpolate(func, deg - 1)).to(device)
            # tmp = np.linspace(-1,1,10000)
            # print(np.mean((np.polynomial.chebyshev.chebval(tmp, self.coeffs[i].cpu().numpy()) - func(tmp))**2))
        self.deg = deg
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

    def normalSample(self):
        self.X = torch.nn.init.normal_(self.X)

    @torch.no_grad()
    def produceSample(self):
        self.X = torch.zeros((self.samples, self.m)).to(self.device).detach()
        self.normalSample()
        # M2 = torch.zeros((self.samples, self.m)).to(self.device).detach()
        # M2 = M2.new_tensor(self.X).to(self.device).detach()
        M2 = self.X.clone().detach()
        M1 = self.matMat(M2)
        # print(torch.linalg.norm(self.X, dim=1).mean())
        # print(torch.linalg.norm(M2, dim=1).mean())
        # print(torch.linalg.norm(M1, dim=1).mean())

        # f = torch.zeros((self.samples, 1)).to(self.device).detach()
        f = (self.coeffs[:, 0] * torch.bmm(self.X.view(self.samples, 1, self.m), M2.view(self.samples, self.m, 1)) +
             self.coeffs[:, 1] * torch.bmm(self.X.view(self.samples, 1, self.m), M1.view(self.samples, self.m, 1)))

        for i in range(2, self.deg):
            F = 2.0 * self.matMat(M1) - M2
            # print(torch.linalg.norm(F, dim=1).mean())

            del M2
            M2 = M1
            M1 = F
            # print(M2.shape,M1.shape,F.shape,torch.norm(F,dim=1))
            f  = f + self.coeffs[:, i] * torch.bmm(self.X.view(self.samples, 1, self.m), F.view(self.samples, self.m, 1))
            # if len(self.coeffs) == 1:
            #     print("f_"+str(i), torch.mean(f, dim=(0, 1)))
        return f#torch.matmul(torch.t(self.X),f)

    def estimateAlpha(self):
        t = torch.zeros(self.coeffs.shape[0]).to(self.device)
        draws = 1
        for repeats in range(draws):
            particles = self.produceSample()
            t += particles.mean(dim=(0, 1))
        t /= 1.0 * draws
        return min(self.m, self.n) / t
    def fit(self):
        alphas = self.estimateAlpha()
        x_min = torch.tensor(self.x_min).to(self.device)
        summe = (min(self.m, self.n) / alphas)
        summe += min(self.m, self.n) * torch.log(x_min)
        likelihoods = min(self.m, self.n) * torch.log(alphas) \
                    + min(self.m, self.n) * alphas * torch.log(x_min) \
                    - summe * (1 + alphas)
        # print(alphas)
        # print(*zip(alphas.cpu().numpy(), likelihoods.cpu().numpy()))
        print(alphas[likelihoods.argmax()].item(), self.x_min[likelihoods.argmax()])

class SpikeApproximator():
    def __init__(self,layer, device,shape_in,shape_out):
        self.layer = layer
        self.device = device
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.m = np.prod(shape_out)
        self.n = np.prod(shape_in)

        # return cp.array(y.view(self.m).detach()).astype(cp.float64)
        # print(w, w.device)
        # y = torch.from_numpy(w.astype(np.float32)).view(self.shape_out).to(self.device)
        # print(y.device)
        # print(y)
        # bitch2 = self.H.register_hook(lambda x: print(x.shape))
        # bitch = self.H.register_hook(lambda x: y)
        # z = torch.sum(self.H)
        # z.backward(retain_graph=True)
        # bitch.remove() #get out the way!
        # y = self.layer(self.X.grad)
        # self.X.grad.data.zero_()
    def matVec(self, w):
        # print(w.shape)
        # y = torch.as_tensor(w, device=self.device).float().view(self.shape_out)
        y = w.view(self.shape_out)
        if isinstance(self.layer, torch.nn.Conv2d):
            yy = torch.nn.functional.conv_transpose2d(y, self.layer.weight, None, self.layer.stride, self.layer.padding, 0, self.layer.groups, self.layer.dilation)
            y = torch.nn.functional.conv2d(yy, self.layer.weight, None, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)
        elif isinstance(self.layer, torch.nn.Linear):
            yy = torch.nn.functional.linear(y, self.layer.weight.T, None)
            y = torch.nn.functional.linear(yy, self.layer.weight, None)
        return y.view(self.m, 1).detach()
    def matVecT(self,y):
        # yy = torch.as_tensor(w, device=self.device, dtype=torch.float32).view(self.shape_out)
        y = y.view(self.shape_in)
        yy = torch.nn.functional.conv2d(y, self.layer.weight, None, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)
        x = torch.nn.functional.conv_transpose2d(yy, self.layer.weight, None, self.layer.stride, self.layer.padding, 0, self.layer.groups, self.layer.dilation)
        x = x.view(self.n, 1).detach()#
        # x = cp.array(x)
        return x
        # yy = torch.from_numpy(y.astype(np.float32)).view(self.shape_in).to(self.device)

        #x = x.cpu().numpy()
        # bitch = self.H.register_hook(lambda x: yy)
        # z = torch.sum(self.H)
        # z.backward(retain_graph=True)
        # bitch.remove() #get out the way!
        
        # # print(x)
        # return x
    def estimateSpikes(self,budget):
        #self.X = torch.autograd.Variable(torch.randn(self.shape_in).to(self.device),requires_grad=True)
        #self.H = self.layer(self.X)
        with cp.cuda.Device(0):
            if True or self.m < self.n:
                s = own_eigsh(self.matVec, self.m, k=min(self.m - 1, budget))
                # s = eigsh(
                #     LinearOperator((self.m,self.m), matvec=self.matVec),
                #     k=min(self.m - 1, budget),
                #     which="LM",
                #     return_eigenvectors=False,
                # )
                # print("logpcg")
                # s2, _ = lobpcg(
                #     LinearOperator((self.m, self.m), matvec=self.matVec),
                #     cp.random.ranf((self.m, budget)),
                #     maxiter=1000
                # )
            else:
                WERENEVERHERE
                s = own_eigsh(self.matVecT, self.n, k=min(self.m - 1, budget))
                # s = eigsh(
                    # LinearOperator((self.n, self.n), matvec=self.matVecT),
                    # k=min(self.n - 1,budget),
                    # which="LM",
                    # return_eigenvectors=False,
                    # ncv=4*budget)
        # del self.X
        # del self.H
        return s.cpu().numpy()



# def matvecT(v):
#     return W.T @ (W @v.reshape(-1,1))
# import cupy as cp
# from cupyx.scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg import eigsh as seigsh
# from scipy.sparse.linalg import LinearOperator as SLinearOperator
# W = cp.eye(100)[:,:50] + cp.random.randn(100,50).astype(cp.float32)
# W = W @ W.T

# def matvec(v):
#     return W @ v.reshape(-1, 1)
# def smatvec(v):
#     return W.get() @ v.reshape(-1, 1)
# def tmatvec(v):
#     return torch.as_tensor(W, device="cuda", dtype=torch.float32) @ v.reshape(-1, 1)
# for k in range(1, 40):
#     print(k,
#           eigsh(LinearOperator((100, 100), matvec=matvec), k=k, return_eigenvectors=False, which="LM")[-1],
#           seigsh(SLinearOperator((100, 100), matvec=smatvec), k=k, return_eigenvectors=False, which="LM"),
#           own_eig(tmatvec, 100, k=k).cpu().numpy()
#     )

# asdfa

from torchvision.models import resnet18, vgg16
from torchinfo import summary
import powerlaw
r = vgg16(pretrained=True)
summ = summary(r, (1, 3, 64, 64), col_names=("input_size", "output_size"))
for l in summ.summary_list[2:]:
    input_size = l.input_size
    output_size = l.output_size

    l = l.module
    if not isinstance(l, (torch.nn.Linear, )):#torch.nn.Conv2d, 
        continue
    print(input_size, output_size)
    s = SpikeApproximator(l, "cuda", input_size, output_size)
    print(l)
    for budget in [1]:
        S = s.estimateSpikes(budget)
        print(S)
        # S = np.sqrt(np.abs(S))
        A = AlohaEstimator(l, "cuda", input_size, output_size, None, S.item(), 16)
        A.fit()
        if isinstance(l,torch.nn.Linear):
            s = torch.linalg.svdvals(l.weight).detach().cpu().numpy()  ** 2
            # print(s)
            P = powerlaw.Fit(s)
            print(P.alpha, P.xmin)

            A = AlohaEstimator(l, "cuda", input_size, output_size, P.xmin, S.item(), 128)
            A.fit()
            def func(x):
                return (((x + 1) * (0.5 * x_max)) > x_min) * np.nan_to_num((np.log((x + 1) * (0.5 * x_max) / x_min)))
            print("exact func", func(s).sum())

            print("cheby func", np.polynomial.chebyshev.chebval(s, A.coeffs[0].cpu().numpy()).sum())

        # # print(S)
        # P = results = powerlaw.Fit(S) #xmin=(S[-1] + np.linalg.svd(l.weight.detach().cpu().numpy().reshape(l.weight.shape[0], -1), compute_uv=False).min()) / 2
        # print(P.alpha, S[-1], S[0])





# l = r.layer2[0].conv1
# l.bias = None
# # l = torch.nn.Conv2d(4, 6, 3, stride=1, padding=1, bias=None).cuda()
# print(l.weight.shape)

# print(s.m, s.n)
# # S = s.estimateSpikes(128)
# # print(np.sqrt(np.abs(S))[::-1])
# # print(np.sqrt(S2)[::-1])
# print(np.linalg.svd(l.weight.detach().cpu().numpy().reshape(l.weight.shape[0], -1), compute_uv=False))

# # print(np.linalg.norm(s.U[:,-1]), torch.linalg.norm(l(torch.tensor(s.U[:,-1]).cuda().view(1,4,16,16))))
# # W = (s.U[:,0:1] @ s.U[:,0:1].T)
# # print(W.shape)
# # print(W)

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# # fig = plt.figure()
# # plt.imshow(W, interpolation='nearest', cmap=cm.Greys_r)
# # plt.savefig("test.png")
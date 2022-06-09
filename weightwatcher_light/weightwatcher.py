import torch
import numpy as np
from torchinfo import summary
from .linalg import TopEigValsEstimator, AlphaEstimator
import powerlaw

def weightwatcher(model, input_shape, device="cuda", chebyshev_degree=64, num_batches=1, batch_size=4 * 4096, verbose=True):
    summ = summary(model, input_shape, col_names=("input_size", "output_size"), verbose=0)
    layerwise_info = []
    for l in summ.summary_list:
        input_size = l.input_size
        output_size = l.output_size
        l = l.module
        if not isinstance(l, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        if verbose:
            print("Analyzing", l, "as ", f"{np.prod(output_size)}x{np.prod(input_size)}", "linear layer")
        
        s = TopEigValsEstimator(l, device, input_size, output_size)
        spectral_norm = s.estimateEigVals(1).item()
        if verbose:
            print("Spectral Norm is", spectral_norm)

        A = AlphaEstimator(l, device, input_size, output_size, None, spectral_norm, chebyshev_degree)
        A.samples = batch_size
        alpha = A.fit(draws=num_batches).item()
        if verbose:
            print("Alpha  is", alpha, "with x_min", A.xmin, f"(D={A.D})")
            if isinstance(l, torch.nn.Linear):
                s = torch.linalg.svdvals(l.weight).detach().cpu().numpy()  ** 2
                # print(s)
                P = powerlaw.Fit(s, suppress_output=True)
                print("svd + powerlaw produces alpha ", P.alpha, "with x_min", P.xmin, f"(D={P.D})")
                print("n", A.est_n.item(), len(s[s > A.xmin]) )
        layerwise_info.append(dict(
            layer=l,
            spectral_norm=spectral_norm,
            alpha=alpha,
            weighted_alpha=np.log10(spectral_norm) * alpha,
            D=A.D.item(),
            input_size=input_size,
            output_size=output_size,
            N=A.n,
            M=A.m,
        ))
    model_info = { k:np.mean([l[k] for l in layerwise_info]) for k in ["spectral_norm", "alpha", "weighted_alpha", "D"]}
    return dict(layers=layerwise_info, **model_info)


if __name__ == "__main__":
    import torchvision.models as models
    results = []
    for model_cls in [models.vgg11_bn, models.vgg13_bn, models.vgg16_bn, models.vgg19_bn]:
        model = model_cls(pretrained=True).cuda()
        statistics = weightwatcher(model, (1, 3, 32, 32))
        statistics.pop("layers")
        print(statistics)
        results.append(statistics)
    
    for n, r in zip(["VGG11", "VGG13", "VGG16", "VGG19"], results):
        print(n, r)
    #VGG13: {'spectral_norm': 81.13524187528171, 'alpha': 2.0631472881023702, 'weighted_alpha': 3.737522625643188}
    #VGG16: {'spectral_norm': 65.65331041812897, 'alpha': 2.0158966332674026, 'weighted_alpha': 3.43908260343017}
    #VGG19: {'spectral_norm': 53.92422615854364, 'alpha': 2.144058164797331, 'weighted_alpha': 3.489194736805615}
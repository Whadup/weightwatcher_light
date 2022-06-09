import torch
import numpy as np
from torchinfo import summary
from .linalg import TopEigValsEstimator, AlphaEstimator


def weightwatcher(model, input_shape, device="cuda", chebyshev_degree=64, num_batches=5, batch_size=4 * 4096, verbose=True, debug=False):
    summ = summary(model, input_shape, col_names=("input_size", "output_size"), verbose=0)
    layerwise_info = []
    for l in summ.summary_list:
        input_size = l.input_size
        output_size = l.output_size
        l = l.module
        if not isinstance(l, (torch.nn.Linear, torch.nn.Conv2d)):#
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

        layer_info = dict(
            layer=l,
            spectral_norm=spectral_norm,
            alpha=alpha,
            weighted_alpha=np.log10(spectral_norm) * alpha,
            D=A.D.item(),
            xmin=A.xmin.item(),
            input_size=input_size,
            output_size=output_size,
            N=A.n,
            M=A.m,
        )

        if verbose:
            print("Alpha  is", alpha, "with x_min", A.xmin, f"(D={A.D})")
        if debug and isinstance(l, torch.nn.Linear):
            import powerlaw
            s = torch.linalg.svdvals(l.weight).detach().cpu().numpy()  ** 2
            P = powerlaw.Fit(s, suppress_output=True)
            layer_info["alpha_svd"] = P.alpha
            layer_info["xmin_svd"] = P.xmin
            layer_info["D_svd"] = P.D

            print("svd + powerlaw produces alpha ", P.alpha, "with x_min", P.xmin, f"(D={P.D})")
            print("n", A.est_n.item(), len(s[s > A.xmin]) )
        elif debug and min(A.n, A.m) < 32000:
            import powerlaw
            s = s.estimateEigVals(min(A.n, A.m))
            P = powerlaw.Fit(s, suppress_output=True)
            print("svd + powerlaw produces alpha ", P.alpha, "with x_min", P.xmin, f"(D={P.D})")
            layer_info["alpha_svd"] = P.alpha
            layer_info["xmin_svd"] = P.xmin
            layer_info["D_svd"] = P.D
        layerwise_info.append(layer_info)
    model_info = { k:np.nanmean([l.get(k, np.nan) for l in layerwise_info]) for k in  layerwise_info[0].keys() if isinstance(layerwise_info[0][k], float)}
    return dict(layers=layerwise_info, **model_info)

import torch
import numpy as np
from torchinfo import summary
from .linalg import TopEigValsEstimator, AlphaEstimator


def weightwatcher(model, input_shape, device="cuda", chebyshev_degree=256, num_batches=8, batch_size=1024):
    summ = summary(model, input_shape, col_names=("input_size", "output_size"), verbose=0)
    layerwise_info = []
    for l in summ.summary_list:
        input_size = l.input_size
        output_size = l.output_size
        print(l)
        l = l.module
        if not isinstance(l, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        
        s = TopEigValsEstimator(l, device, input_size, output_size)
        spectral_norm = s.estimateEigVals(1).item()

        A = AlohaEstimator(l, device, input_size, output_size, None, spectral_norm, chebyshev_degree)
        alpha = A.fit()
        layerwise_info.append(dict(
            layer=l,
            spectral_norm=spectral_norm,
            alpha=alpha,
            weighted_alpha=np.log10(spectral_norm) * alpha,
            input_size=input_size,
            output_size=output_size,
            N=A.n,
            M=A.m,
        ))
    model_info = { k:np.mean([l[k] for l in layerwise_info]) for k in layerwise_info[0].keys() if key!="layer"}
    return dict(layers=layerwise_info, **model_info)


if __name__ == "__main__":
    from torchvision.models import resnet18, vgg16
    model = vgg16(pretrained=True).to("cuda")
    print(weightwatcher(model, (1, 3, 64, 64)))
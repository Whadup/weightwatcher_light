# WeightWatcher *light*

A small reimplementation of the [`weightwatcher`](https://github.com/CalculatedContent/WeightWatcher) project. No affiliation whatsoever.
Why? For fun. Mostly. And to toy around with Chebyshev polynomials for matrices...

## Demo

``` python
# python demo.py

import pandas as pd
import torchvision.models as models
from weightwatcher_light import weightwatcher
if __name__ == "__main__":
    results = []
    for model_cls in [models.vgg11, models.vgg13, models.vgg16, models.vgg19]:
        print(f"======{model_cls.__name__}======")
        model = model_cls(pretrained=True).cuda()
        statistics = weightwatcher(model, (1, 3, 32, 32), verbose=False, debug=False)
        print(pd.DataFrame(statistics.pop("layers")).to_markdown())
        results.append(statistics)
    
    for n, r in zip(["VGG11", "VGG13", "VGG16", "VGG19"], results):
        print(n, r)
```

|    | layer                                                               |   spectral_norm |    alpha |   weighted_alpha |         D |       xmin | input_size      | output_size      |     N |     M |
|---:|:--------------------------------------------------------------------|----------------:|---------:|-----------------:|----------:|-----------:|:----------------|:-----------------|------:|------:|
|  0 | Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))    |        252.795  | 11.9232  |         28.6487  | 0.105255  | 202.4      | [1, 3, 32, 32]  | [1, 64, 32, 32]  |  3072 | 65536 |
|  1 | Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  |         89.7801 |  1.72747 |          3.37405 | 0.0873823 |   2.69943  | [1, 64, 16, 16] | [1, 128, 16, 16] | 16384 | 32768 |
|  2 | Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |         73.0979 |  1.77617 |          3.31062 | 0.064562  |   1.24577  | [1, 128, 8, 8]  | [1, 256, 8, 8]   |  8192 | 16384 |
|  3 | Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |         81.6822 |  4.6921  |          8.9719  | 0.046055  |  33.477    | [1, 256, 8, 8]  | [1, 256, 8, 8]   | 16384 | 16384 |
|  4 | Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |         89.7895 |  2.10929 |          4.11992 | 0.0650091 |   2.7608   | [1, 256, 4, 4]  | [1, 512, 4, 4]   |  4096 |  8192 |
|  5 | Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |        176.504  |  3.06774 |          6.89246 | 0.0357556 |  12.602    | [1, 512, 4, 4]  | [1, 512, 4, 4]   |  8192 |  8192 |
|  6 | Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |         51.9428 |  3.84261 |          6.5921  | 0.0478162 |  10.083    | [1, 512, 2, 2]  | [1, 512, 2, 2]   |  2048 |  2048 |
|  7 | Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |        130.978  |  3.23253 |          6.84391 | 0.0387834 |   5.59936  | [1, 512, 2, 2]  | [1, 512, 2, 2]   |  2048 |  2048 |
|  8 | Linear(in_features=25088, out_features=4096, bias=True)             |         34.7824 |  2.38513 |          3.67634 | 0.0289069 |   0.431471 | [1, 25088]      | [1, 4096]        | 25088 |  4096 |
|  9 | Linear(in_features=4096, out_features=4096, bias=True)              |         60.2786 |  2.22953 |          3.96892 | 0.0336329 |   0.417442 | [1, 4096]       | [1, 4096]        |  4096 |  4096 |
| 10 | Linear(in_features=4096, out_features=1000, bias=True)              |         58.7861 |  2.77698 |          4.91324 | 0.0201149 |   5.53593  | [1, 4096]       | [1, 1000]        |  4096 |  1000 |

## Major Differences

Beside the much smaller number of features, the most signficicant difference is how we estimate the eigenvalues of convolution layers:
Per default, `weightwatcher` treats convolution kernels of size $w \times h$ as $w\cdot h$ separate matrices, and estimates their eigenvalues. 
Instead, we treat both linear layers and convolution layers as a linear operators that map from one vector space to another and we estimate the eigenvalues of these operators.
For large input images, the rank of these operators can be larger than 100,000. 
This is why we apply an approximation technique based on Hutchinson's trace estimators in conjunction with Chebyshev polynomials to obtain the best powerlaw fit.

## Mathematical Details

Estimating powerlaws of the eigenvalues $\lambda_1,\ldots,\lambda_n$ requires estimation of two parameters: $\alpha$, the scaling parameter, and $x_m$, the lower bound.
Given the lower bound, the maximum-likelihood estimator for the scale is 
$$\hat \alpha = 1 + \frac{\sum\limits_{i=1}^n 1[\lambda_i \geq x_m]}{ \sum\limits_{i=1}^n 1[\lambda_i \geq x_m]\log \frac {\lambda_i}{x_m}}.$$
We estimate the sums in the nominator and denominator using Hutchinson's trace estimators and Chebyshev polynomials. First, recall that the trace can be written as an expectation, the so-called Hutchinson trace estimator

$$
\mathrm{tr}(W) = \sum\limits_{i=1}^n \lambda_i = \mathbb E_{x \sim \mathcal N(0,I)} x^TWx
$$

and hence we can estimate the trace using a finite number of samples.

For a matrix function $f(W)$, we can analogously estimate 

$$
\mathrm{tr}(f(W)) = \sum\limits_{i=1}^n f(\lambda_i) = \mathbb E_{x \sim \mathcal N(0,I)} x^Tf(W)x,
$$

and in case of the denominator we are interested in the function family 

$$
f(\lambda) = 1[\lambda \geq x_m]\log \frac {\lambda}{x_m}
$$

for the set of possible lower bounds $x_m$.
We neither want to compute all $\lambda_i$ numerically, as $W$ is very high-dimensional, nor can we compute $f(W)$. Instead, we approximate $f(\lambda)$ using Chebyshev polynomials. Consequently, we can approximate $f(W)$ using the matrix Chebyshev polynomial base and the coefficient approximated on $f(\lambda)$. 
From an implementation viewpoint, in conjunction with the trace estimator,we require no knowledge of the matrix $W$; it suffices to know how to compute the matrix-vector products $Wx$ and $W^Ty$. In case of convolution layers, these are the forward- and backward-pass implementations of convolution.

We apply a similar technique to estimate the number of eigenvalues larger than $x_m$, which we also use to estimate the empirical cumulative distribution needed to estimate the lower bound $x_m$ using the KS-method proposed by Clauset et al.

You can read more about these Matrix Chebyshev techniques in 
1. Napoli, E. Di, Polizzi, E., & Saad, Y. (2016). Efficient estimation of eigenvalue counts in an interval. Numerical Linear Algebra with Applications. 
2. Adams, R. P., Pennington, J., Johnson, M. J., Smith, J., Ovadia, Y., Patton, B., & Saunderson, J. (2018). Estimating the Spectral Density of Large Implicit Matrices (Vol. 46).

## Issues
Estimating $x_m$ seems to be pretty brittle, but has a huge influence on the estimated $\alpha$. There seem to be many solutions to the KS-estimator that are about equally good in terms of the distance between empirical and estimated fit. Given the approximation errors, we may chose one that looks optimal merely because of the uncertainty.
This is not necessarily a problem of this approach, I suspect that the `powerlaw` package has similar issues. Honestly, the motivation for the KS-estimator is kind of heuristic, I think looking for alternatives is a good idea.

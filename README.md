# WeightWatcher *light*

A small reimplementation of the [`weightwatcher`](https://github.com/CalculatedContent/WeightWatcher) project. No affiliation whatsoever.
Why? For fun. Mostly. And to toy around with Chebyshev polynomials for matrices...

## Demo

``` python
# python demo.py

import torchvision.models as models
from weightwatcher_light import weightwatcher
if __name__ == "__main__":
    results = []
    for model_cls in [models.vgg11, models.vgg13, models.vgg16, models.vgg19]:
        print(f"======{model_cls.__name__}======")
        model = model_cls(pretrained=True).cuda()
        statistics = weightwatcher(model, (1, 3, 32, 32), verbose=True, debug=True)
        results.append(statistics)
    
    for n, r in zip(["VGG11", "VGG13", "VGG16", "VGG19"], results):
        print(n, r)
```

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
f(\lambda) = 1[\lambda_i \geq x_m]\log \frac {\lambda_i}{x_m}
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

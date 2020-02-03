# GPFlux.jl

A new Gaussian process package, compare to existing ones (GaussianProcesses.jl, Stheno.jl), it use Zygote to compute derivatives w.r.t model parameters, and is also compatible with Flux, which enables user to integrate deep neural network into Gaussian process model (e.g. use neural network as mean function or kernel function).

This package is still under development, **suggestions**, **bug report** and **pull request** are welcome :)

## Installation
To install GPFlux.jl, please run the following code in a Julia REPL:
```julia
] add GPFlux
```

## Brief introduction to GP
Gaussian processe is a powerful algorithm in statistical machine learning and probabilistic modelling, it models the underlying distribution of a dataset by a prior belief ( which is a parametrized multivariate normal distribution ) and a Gaussian likelihood, learning is done by maximizing the log likelihood (MLE), which is tractable for Gaussian process. Gaussian process is widely used in surrogate function modelling, geostatitics, pattern recognition, etc.


## Examples
For details, please see notebooks.
### Simple regression

### Time series prediction


## Usage
Gaussian process is determined by a mean function and a kernel function, they can be specified in GPFlux as follows
```julia
# mean function
c = [0.0] # mean constant
zero_mean = ConstantMean(c)
# square exponential kernel
ll = [0.0] # length scale in log scale
lσ = [0.0] # scaling factor in log scale
se_kernel = IsoGaussKernel(ll, lσ)
# build Gauss process
lnoise = [-2.0] # noise in log scale
gp = GaussProcess(zero_mean, se_kernel, lnoise)
```
The parameters in the above `gp` model are `c`, `ll`, `lσ` and `lnoise`, one can extract all parameters by:
```julia
ps = params(gp)
```
Given data `X`,`y`, one can compute the negative log likelihood and it's gradient w.r.t all the parameters by:
```julia
negloglik(gp, X, y) # (X, y) is the dataset
gradient(()->negloglik(gp, X, y), ps)
```
which are straight forward if you are familiar with Flux and Zygote.


One can also build composite kernel by using `ProductCompositeKernel` and `AddCompositeKernel`( **Note: AD works for arbitrary composite kernels** ).
```julia
se_kernel = IsoGaussKernel(ll, lσ)
per_kernel = IsoPeriodKernel(lp, ll, lσ)
se_mul_periodic_kernel = ProductCompositeKernel((se_kernel, per_kernel))
se_add_periodic_kernel = AddCompositeKernel((se_kernel, per_kernel))

params(se_mul_periodic_kernel) # provide parameters of se_mul_periodic_kernel
params(se_add_periodic_kernel) # provide parameters of se_add_periodic_kernel
```

The most significant feature of GPFlux is that it allows to use Flux's neural network to build mean function and Neural Kernel Network (NKN) to build kernel function ( *Warning: NKN is still in progress* ), computing negative log likelihood and it's gradient is same as above cases.
```julia
# build the mean function with neural network using Flux
nn_mean = Chain(Dense(5, 10, relu), Dense(10, 1))
# build the kernel function with neural kernel network
nkn = Chain(Primitive(se_kernel, per_kernel), Linear(2, 4), z->Product(z, step=4), z->reshape(z, N, N)) # N is the number of samples in dataset
# build GP
nn_gp = GaussProcess(nn_mean, nkn, lnoise)

# compute negative log likelihood and gradient
negloglik(nn_gp, X, y)
gradient(()->negloglik(nn_gp, X, y), params(nn_gp))
```

Once we have negative log likelihood and gradients, we can either use Optim.jl or Flux's optimizers to do optimization.

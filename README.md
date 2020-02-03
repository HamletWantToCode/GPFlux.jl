# GPFlux.jl

A new Gaussian Process (GP) package, compare to the existing GP packages (GaussianProcesses.jl, Stheno.jl), it utilizes Zygote.jl's remove mode automatic differentiation (AD) to calculate derivatives with respect to GP parameters, and is compatible with Flux.jl, this allows user to use deep neural network in GP model (e.g. use neural network as mean function or kernel function).

This package is still under development, **suggestions**, **bug report** as well as **pull request** are welcome :)

## Installation
To install GPFlux.jl, please run following code in a Julia REPL:
```julia
] add GPFlux
```

## Brief introduction to GP
Gaussian processe is a powerful algorithm in statistical machine learning and probabilistic modelling, it models unknown function by samples from multivariate normal distribution ( *prior* distribution), and uses Gaussian *likelihood* to integrate dataset to the prior distribution. Learning GP model can be done by maximizing GP's log likelihood, which is tractable. GP is widely used in surrogate function modelling, geostatitics, pattern recognition, etc.


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
l\sigma = [0.0] # scaling factor in log scale
se_kernel = IsoGaussKernel(ll, l\sigma)
# build Gauss process
lnoise = [-2.0] # noise in log scale
gp = GaussProcess(zero_mean, se_kernel, lnoise)
```
The parameters in the above GP model are constant in mean function, length scale, scaling factor in kernel and noise, you can extract all parameters by:
```julia
ps = params(gp)
```
and given data, you can compute negative log likelihood and it's gradient w.r.t all the parameters by:
```julia
negloglik(gp, X, y) # (X, y) is the dataset
gradient(()->negloglik(gp, X, y), ps)
```
which are straight forward if you are familiar with Flux.jl and Zygote.jl.


You can also build composite kernel by using `ProductCompositeKernel` and `AddCompositeKernel`, and **AD for arbitrary composite kernels is supported**.
```julia
se_kernel = IsoGaussKernel(ll, l\sigma)
per_kernel = IsoPeriodKernel(lp, ll, l\sigma)
se_mul_periodic_kernel = ProductCompositeKernel((se_kernel, per_kernel))
se_add_periodic_kernel = AddCompositeKernel((se_kernel, per_kernel))

params(se_mul_periodic_kernel) # provide parameters of se_mul_periodic_kernel
params(se_add_periodic_kernel) # provide parameters of se_add_periodic_kernel
```

The most significant feature of GPFlux is that it can use Flux's neural network to build mean function and Neural Kernel Network (NKN) to build kernel function ( *Warning: this properties is still in progress* ).
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


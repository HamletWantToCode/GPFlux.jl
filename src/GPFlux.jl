module GPFlux

using LinearAlgebra
using BackwardsLinalg
using Flux
using Flux: @functor, params
import Base: reset
import Distributions: MvNormal


export GaussProcess, MvNormal, negloglik, predict, params, dispatch!, flatten_params

export ConstantMean, SimpleNeuralNetworkMean 
			 
export ArdGaussKernel, IsoGaussKernel,
			 ArdRQKernel, IsoRQKernel,
			 ArdLinearKernel, IsoLinearKernel,
			 IsoPeriodKernel

export ProductCompositeKernel, AddCompositeKernel,
			 SE_mul_PeriodKernel, Lin_mul_LinKernel, Period_mul_LinKernel, SE_mul_LinKernel,
			 SE_add_PeriodKernel

export Chain, Primitive, Linear, Product, allProduct, allSum 

export norm2_metric, square_metric, inner_prod_metric

export gradient_check, model_gradient_check, reset


include("gp.jl")
include("mean.jl")
include("layers.jl")
include("kernels/kernels.jl")
include("metrics/metrics.jl")
include("zygote.jl")

end # module

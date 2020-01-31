module DiffGP

using LinearAlgebra
using BackwardsLinalg
using Flux
using Flux: @functor, functor, params
import Base: reset

export GaussProcess, negloglik, 
			 ConstantMean, SimpleNeuralNetworkMean, 
			 GaussKernel, IsoGaussKernel,
			 RQKernel, IsoRQKernel,
			 IsoPeriodKernel, LinearKernel,
			 params

export Primitive, Linear, Product, positive 

export norm2_metric, square_metric, inner_prod_metric

export gradient_check, model_gradient_check, reset



# Gauss process 
struct GaussProcess{MT, KT}
	mean::MT
	kernel::KT
end
@functor GaussProcess


# loss function 
function negloglik(gp::GaussProcess, x, y)
	μ = reshape(gp.mean(x), size(y))
  Σ = gp.kernel(x)

  d = length(y)
  L = BackwardsLinalg.cholesky(Σ)
	ȳ = y .- μ
	z = L \ ȳ
	z̄ = L' \ z
  0.5*dot(ȳ, z̄) + (d/2.0)*log(2π) + logdet(L)
end


include("mean.jl")
include("kernel.jl")
include("layers.jl")
include("norm2_metric.jl")
include("square_metric.jl")
include("inner_prod_metric.jl")
include("zygote.jl")

end # module

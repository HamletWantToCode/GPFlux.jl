"""
composite kernel
"""

import Flux: functor


struct ProductCompositeKernel{T<:Tuple}
	kernels::T
end
ProductCompositeKernel(ks...) = ProductCompositeKernel{typeof(ks)}(ks) 
functor(pck::ProductCompositeKernel) = pck.kernels, ks->ProductCompositeKernel(ks...)
function (PCK::ProductCompositeKernel)(x; λ=1e-6)
	N = size(x, 2)
	K = Chain(Primitive(PCK.kernels...), allProduct, z->reshape(z, N, N))
	K(x) + Diagonal(λ*ones(N))
end


struct AddCompositeKernel{T<:Tuple}
	kernels::T
end
AddCompositeKernel(ks...) = AddCompositeKernel{typeof(ks)}(ks)
functor(ack::AddCompositeKernel) = ack.kernels, ks->AddCompositeKernel(ks...)
function (ACK::AddCompositeKernel)(x; λ=1e-6)
	N = size(x, 2)
	K = Chain(Primitive(ACK.kernels...), allSum, z->reshape(z, N, N))
	K(x) + Diagonal(λ*ones(N))
end



"""
common composite kernels
"""
const SE_mul_PeriodKernel = ProductCompositeKernel{Tuple{IsoGaussKernel, IsoPeriodKernel}}
const Lin_mul_LinKernel = ProductCompositeKernel{Tuple{IsoLinearKernel, IsoLinearKernel}}
const Period_mul_LinKernel = ProductCompositeKernel{Tuple{IsoPeriodKernel, IsoLinearKernel}}
const SE_mul_LinKernel = ProductCompositeKernel{Tuple{IsoGaussKernel, IsoLinearKernel}}

const SE_add_PeriodKernel = AddCompositeKernel{Tuple{IsoGaussKernel, IsoPeriodKernel}}


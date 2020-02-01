"""
composite kernel
"""
struct ProductCompositeKernel{KT1, KT2}
	K1::KT1
	K2::KT2
end
@functor ProductCompositeKernel
function (PCK::ProductCompositeKernel)(x; λ=1e-6)
	N = size(x, 2)
	K = Chain(Primitive(PCK.K1, PCK.K2), Product, z->reshape(z, N, N))
	K(x) + Diagonal(λ*ones(N))
end


struct AddCompositeKernel{KT1, KT2, LT}
	K1::KT1
	K2::KT2
	Lin::LT
end
@functor AddCompositeKernel
function (ACK::AddCompositeKernel)(x; λ=1e-6)
	N = size(x, 2)
	K = Chain(Primitive(ACK.K1, ACK.K2), ACK.Lin, z->reshape(z, N, N))
	K(x) + Diagonal(λ*ones(N))
end



"""
common composite kernels
"""
const SE_mul_PeriodKernel = ProductCompositeKernel{IsoGaussKernel, IsoPeriodKernel}
const Lin_mul_LinKernel = ProductCompositeKernel{IsoLinearKernel, IsoLinearKernel}
const Period_mul_LinKernel = ProductCompositeKernel{IsoPeriodKernel, IsoLinearKernel}
const SE_mul_LinKernel = ProductCompositeKernel{IsoGaussKernel, IsoLinearKernel}

const SE_add_PeriodKernel = AddCompositeKernel{IsoGaussKernel, IsoPeriodKernel, Linear}


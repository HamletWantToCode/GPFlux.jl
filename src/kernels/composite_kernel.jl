"""
general composite kernel
"""

const ProductCompositeKernel = NeuralKernelNetwork{Tuple{Primitive, typeof(allProduct)}}

const AddCompositeKernel = NeuralKernelNetwork{Tuple{Primitive, typeof(allSum)}}


"""
special composite kernels
"""
function SE_mul_PeriodKernel(k1::IsoGaussKernel, k2::IsoPeriodKernel)
	pk = Primitive(k1, k2)
	ProductCompositeKernel(pk, allProduct)
end

function Lin_mul_LinKernel(k1::IsoLinearKernel, k2::IsoLinearKernel)
	pk = Primitive(k1, k2)
	ProductCompositeKernel(pk, allProduct)
end

function Period_mul_LinKernel(k1::IsoPeriodKernel, k2::IsoLinearKernel)
	pk = Primitive(k1, k2)
	ProductCompositeKernel(pk, allProduct)
end

function SE_mul_LinKernel(k1::IsoGaussKernel, k2::IsoLinearKernel)
	pk = Primitive(k1, k2)
	ProductCompositeKernel(pk, allProduct)
end

function SE_add_PeriodKernel(k1::IsoGaussKernel, k2::IsoPeriodKernel)
	pk = Primitive(k1, k2)
	AddCompositeKernel(pk, allSum)
end


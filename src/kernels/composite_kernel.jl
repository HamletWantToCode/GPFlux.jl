"""
general composite kernel
"""
const ProductCompositeKernel = NeuralKernelNetwork{Tuple{Primitive, typeof(allProduct)}}
ProductCompositeKernel(K::Vararg{AbstractKernel, N}) where {N} = ProductCompositeKernel(Primitive(K...), allProduct)

const AddCompositeKernel = NeuralKernelNetwork{Tuple{Primitive, typeof(allSum)}}
AddCompositeKernel(K::Vararg{AbstractKernel, N}) where {N} = AddCompositeKernel(Primitive(K...), allSum)


"""
special composite kernels
"""
SE_mul_PeriodKernel(k1::IsoGaussKernel, k2::IsoPeriodKernel) = ProductCompositeKernel(k1, k2)
Lin_mul_LinKernel(k1::IsoLinearKernel, k2::IsoLinearKernel) = ProductCompositeKernel(k1, k2)
Period_mul_LinKernel(k1::IsoPeriodKernel, k2::IsoLinearKernel) = ProductCompositeKernel(k1, k2)
SE_mul_LinKernel(k1::IsoGaussKernel, k2::IsoLinearKernel) = ProductCompositeKernel(k1, k2)
SE_add_PeriodKernel(k1::IsoGaussKernel, k2::IsoPeriodKernel) = AddCompositeKernel(k1, k2)


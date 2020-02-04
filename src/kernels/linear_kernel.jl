"""
Linear Kernel
"""
struct ArdLinearKernel{T, VT<:AbstractVector{T}} <: AbstractKernel
	lσb::VT
	lσv::VT
	c::VT
end
@functor ArdLinearKernel
function (LK::ArdLinearKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(LK.c) == size(x, 1) && length(LK.lσv) == size(x, 1) || throw(DimensionMismatch("size of offset parameter should be the same as number of features"))
	n = size(x, 2)
	shifted_x = x .- LK.c
	scaled_shifted_x = @. shifted_x*exp(-LK.lσv)
	d = inner_prod_metric(scaled_shifted_x)
	σb_square = exp(2*LK.lσb[1])
	(@. d+σb_square) + Diagonal(λ*ones(n))
end
function (LK::ArdLinearKernel{T, VT})(x, xo) where {T, VT}
	shifted_x = x .- LK.c
	scaled_shifted_x = @. shifted_x*exp(-LK.lσv)
	shifted_xo = xo .- LK.c
	scaled_shifted_xo = @. shifted_xo*exp(-LK.lσv)
	d = inner_prod_metric(scaled_shifted_x, scaled_shifted_xo)
	σb_square = exp(2*LK.lσb[1])
	d .+ σb_square
end

function ArdLinearKernel(n_features::Int)
	lσb = rand(1)
	lσv = rand(n_features)
	c  = rand(n_features)
	ArdLinearKernel(lσb, lσv, c)
end

reset(LK::ArdLinearKernel) = ArdLinearKernel(length(LK.c))


## iso kernel
struct IsoLinearKernel{T, VT<:AbstractVector{T}} <: AbstractKernel
	lσv::VT
end
@functor IsoLinearKernel
function (ILK::IsoLinearKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x .* exp(-ILK.lσv[1])
	d = inner_prod_metric(scaled_x)
	d + Diagonal(λ*ones(n))
end
function (ILK::IsoLinearKernel{T, VT})(x, xo) where {T, VT}
	scaled_x = x .* exp(-ILK.lσv[1])
	scaled_xo = xo .* exp(-ILK.lσv[1])
	d = inner_prod_metric(scaled_x, scaled_xo)
	d
end

function IsoLinearKernel()
	lσv = rand(1)
	IsoLinearKernel(lσv)
end

reset(IKL::IsoLinearKernel) = IsoLinearKernel()


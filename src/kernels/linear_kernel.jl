"""
Linear Kernel
"""
struct ArdLinearKernel{T, VT<:AbstractVector{T}}
	σb::VT
	σv::VT
	c::VT
end
@functor ArdLinearKernel
function (LK::ArdLinearKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(LK.c) == size(x, 1) && length(LK.σv) == size(x, 1) || throw(DimensionMismatch("size of offset parameter should be the same as number of features"))
	n = size(x, 2)
	shifted_x = x .- LK.c
	scaled_shifted_x = shifted_x ./ LK.σv
	d = inner_prod_metric(scaled_shifted_x)
	σb_square = LK.σb[1]*LK.σb[1]
	(@. d+σb_square) + Diagonal(λ*ones(n))
end
function (LK::ArdLinearKernel{T, VT})(x, xo) where {T, VT}
	shifted_x = x .- LK.c
	scaled_shifted_x = shifted_x ./ LK.σv
	shifted_xo = xo .- LK.c
	scaled_shifted_xo = shifted_xo ./ LK.σv
	d = inner_prod_metric(scaled_shifted_x, scaled_shifted_xo)
	σb_square = LK.σb[1]*LK.σb[1]
	d+σb_square
end

function ArdLinearKernel(n_features::Int)
	σb = rand(1)
	σv = rand(n_features)
	c  = rand(n_features)
	ArdLinearKernel(σb, σv, c)
end

reset(LK::ArdLinearKernel) = ArdLinearKernel(length(LK.c))


## iso kernel
struct IsoLinearKernel{T, VT<:AbstractVector{T}}
	σv::VT
end
@functor IsoLinearKernel
function (ILK::IsoLinearKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x ./ ILK.σv
	d = inner_prod_metric(scaled_x)
	d + Diagonal(λ*ones(n))
end
function (ILK::IsoLinearKernel{T, VT})(x, xo) where {T, VT}
	scaled_x = x ./ ILK.σv
	scaled_xo = xo ./ ILK.σv
	d = inner_prod_metric(scaled_x, scaled_xo)
	d
end

function IsoLinearKernel()
	σv = rand(1)
	IsoLinearKernel(σv)
end

reset(IKL::IsoLinearKernel) = IsoLinearKernel()


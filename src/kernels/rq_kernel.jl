"""
Rational quardratic kernel
"""
struct ArdRQKernel{T, VT<:AbstractVector{T}}
	ll::VT
	lα::VT
	lσ::VT
end
@functor ArdRQKernel
function (RQK::ArdRQKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(RQK.ll) == size(x, 1) || throw(DimensionMismatch("size of length scale parameter should be the same as number of features"))
	n = size(x, 2)
	scaled_x = @. x*exp(-RQK.ll)
	d = square_metric(scaled_x)
	σ_square = exp(2*RQK.lσ[1])
	σ_square*power(1.0 .+ 0.5*exp(-RQK.lα[1])*d, -exp(RQK.lα[1])) + Diagonal(λ*ones(n))
end
function (RQK::ArdRQKernel{T, VT})(x, xo) where {T, VT}
	σ_square = exp(2*RQK.lσ[1])
	scaled_x = @. x*exp(-RQK.ll)
	scaled_xo = @. xo*exp(-RQK.ll)
	d = square_metric(scaled_x, scaled_xo)
	σ_square*power(1.0 .+ 0.5*exp(-RQK.lα[1])*d, -exp(RQK.lα[1]))
end

function ArdRQKernel(n_features::Int)
	l = rand(n_features)
	lα = rand(1)
	lσ = rand(1)
	ArdRQKernel(ll, lα, lσ)
end

reset(RQK::ArdRQKernel) = ArdRQKernel(length(RQK.ll))

## derived kernel
struct IsoRQKernel{T, VT<:AbstractVector{T}}
	ll::VT
	lα::VT
	lσ::VT
end
@functor IsoRQKernel
function (IsoRQK::IsoRQKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x .* exp(-IsoRQK.ll[1])
	d = square_metric(scaled_x)
	σ_square = exp(2*IsoRQK.lσ[1])
	σ_square*power(1.0 .+ 0.5*exp(-IsoRQK.lα[1])*d, -exp(IsoRQK.lα[1])) + Diagonal(λ*ones(n))
end
function (IsoRQK::IsoRQKernel{T, VT})(x, xo) where {T, VT}
	σ_square = exp(2*IsoRQK.lσ[1])
	scaled_x = x .* exp(-IsoRQK.ll[1])
	scaled_xo = xo .* exp(-IsoRQK.ll[1])
	d = square_metric(scaled_x, scaled_xo)
	σ_square*power(1.0 .+ 0.5*exp(-IsoRQK.lα[1])*d, -exp(IsoRQK.lα[1]))
end

function IsoRQKernel()
	ll = rand(1)
	lα = rand(1)
	lσ = rand(1)
	IsoRQKernel(ll, lα, lσ)
end

reset(IsoRQK::IsoRQKernel) = IsoRQKernel()



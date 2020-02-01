"""
Rational quardratic kernel
"""
struct ArdRQKernel{T, VT<:AbstractVector{T}}
	l::VT
	α::VT
	σ::VT
end
@functor ArdRQKernel
function (RQK::ArdRQKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(RQK.l) == size(x, 1) || throw(DimensionMismatch("size of length scale parameter should be the same as number of features"))
	n = size(x, 2)
	scaled_x = x ./ RQK.l
	d = square_metric(scaled_x)
	σ_square = RQK.σ[1]*RQK.σ[1]
	(@. σ_square*exp((-RQK.α[1])*log(1.0 + 0.5*d/RQK.α[1]))) + Diagonal(λ*ones(n))
end
function (RQK::ArdRQKernel{T, VT})(x, xo) where {T, VT}
	σ_square = RQK.σ[1]*RQK.σ[1]
	scaled_x = x ./ RQK.l
	scaled_xo = xo ./ RQK.l
	d = square_metric(scaled_x, scaled_xo)
	@. σ_square*exp((-RQK.α[1])*log(1.0 + 0.5*d/RQK.α[1]))
end

function ArdRQKernel(n_features::Int)
	l = rand(n_features)
	α = rand(1)
	σ = rand(1)
	ArdRQKernel(l, α, σ)
end

reset(RQK::ArdRQKernel) = ArdRQKernel(length(RQK.l))

## derived kernel
struct IsoRQKernel{T, VT<:AbstractVector{T}}
	l::VT
	α::VT
	σ::VT
end
@functor IsoRQKernel
function (IsoRQK::IsoRQKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x ./ IsoRQK.l[1]
	d = square_metric(scaled_x)
	σ_square = IsoRQK.σ[1]*IsoRQK.σ[1]
	(@. σ_square*exp((-IsoRQK.α[1])*log(1.0 + 0.5*d/IsoRQK.α[1]))) + Diagonal(λ*ones(n))
end
function (IsoRQK::IsoRQKernel{T, VT})(x, xo) where {T, VT}
	σ_square = IsoRQK.σ[1]*IsoRQK.σ[1]
	scaled_x = x ./ IsoRQK.l[1]
	scaled_xo = xo ./ IsoRQK.l[1]
	d = square_metric(scaled_x, scaled_xo)
	@. σ_square*exp((-IsoRQK.α[1])*log(1.0 + 0.5*d/IsoRQK.α[1]))
end

function IsoRQKernel()
	l = rand(1)
	α = rand(1)
	σ = rand(1)
	IsoRQKernel(l, α, σ)
end

reset(IsoRQK::IsoRQKernel) = IsoRQKernel()



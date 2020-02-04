"""
RBF Kernel
	all parameters are in log-scale
"""
struct ArdGaussKernel{T, VT<:AbstractVector{T}} <: AbstractKernel
	ll::VT
	lσ::VT
end
@functor ArdGaussKernel
function (GK::ArdGaussKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(GK.ll) == size(x, 1) || throw(DimensionMismatch("size of length scale parameter should be the same as number of features"))
	n = size(x, 2)
	scaled_x = @. x*exp(-GK.ll) 
	d = square_metric(scaled_x)
	σ_square = exp(2*GK.lσ[1])
	(@. σ_square*exp(-0.5*d)) + Diagonal(λ*ones(n))
end
function (GK::ArdGaussKernel{T, VT})(x, xo) where {T, VT}
	σ_square = exp(2*GK.lσ[1])
	scaled_x = @. x*exp(-GK.ll)
	scaled_xo = @. xo*exp(-GK.ll)
	d = square_metric(scaled_x, scaled_xo)
	@. σ_square*exp(-0.5*d)
end

function ArdGaussKernel(n_features::Int)
	ll = rand(n_features)
	lσ = rand(1)
	ArdGaussKernel(ll, lσ)
end

reset(GK::ArdGaussKernel) = ArdGaussKernel(length(GK.ll))


## iso kernel
struct IsoGaussKernel{T, VT<:AbstractVector{T}} <: AbstractKernel
	ll::VT
	lσ::VT
end
@functor IsoGaussKernel
function (IsoGK::IsoGaussKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x .* exp(-IsoGK.ll[1]) 
	d = square_metric(scaled_x)
	σ_square = exp(2*IsoGK.lσ[1])
	(@. σ_square*exp(-0.5*d)) + Diagonal(λ*ones(n))
end
function (IsoGK::IsoGaussKernel{T, VT})(x, xo) where {T, VT}
	σ_square = exp(2*IsoGK.lσ[1])
	scaled_x = x .* exp(-IsoGK.ll[1])
	scaled_xo = xo .* exp(-IsoGK.ll[1])
	d = square_metric(scaled_x, scaled_xo)
	@. σ_square*exp(-0.5*d)
end

function IsoGaussKernel()
	ll = rand(1)
	lσ = rand(1)
	IsoGaussKernel(ll, lσ)
end

reset(IsoGK::IsoGaussKernel) = IsoGaussKernel()



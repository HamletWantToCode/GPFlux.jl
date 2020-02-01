"""
RBF Kernel
"""
struct ArdGaussKernel{T, VT<:AbstractVector{T}}
	l::VT
	σ::VT
end
@functor ArdGaussKernel
function (GK::ArdGaussKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(GK.l) == size(x, 1) || throw(DimensionMismatch("size of length scale parameter should be the same as number of features"))
	n = size(x, 2)
	scaled_x = x ./ GK.l 
	d = square_metric(scaled_x)
	σ_square = GK.σ[1]*GK.σ[1]
	(@. σ_square*exp(-0.5*d)) + Diagonal(λ*ones(n))
end
function (GK::ArdGaussKernel{T, VT})(x, xo) where {T, VT}
	σ_square = GK.σ[1]*GK.σ[1]
	scaled_x = x ./ GK.l
	scaled_xo = xo ./ GK.l
	d = square_metric(scaled_x, scaled_xo)
	@. σ_square*exp(-0.5*d)
end

function ArdGaussKernel(n_features::Int)
	l = rand(n_features)
	σ = rand(1)
	ArdGaussKernel(l, σ)
end

reset(GK::ArdGaussKernel) = ArdGaussKernel(length(GK.l))


## iso kernel
struct IsoGaussKernel{T, VT<:AbstractVector{T}}
	l::VT
	σ::VT
end
@functor IsoGaussKernel
function (IsoGK::IsoGaussKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x ./ IsoGK.l[1] 
	d = square_metric(scaled_x)
	σ_square = IsoGK.σ[1]*IsoGK.σ[1]
	(@. σ_square*exp(-0.5*d)) + Diagonal(λ*ones(n))
end
function (IsoGK::IsoGaussKernel{T, VT})(x, xo) where {T, VT}
	σ_square = IsoGK.σ[1]*IsoGK.σ[1]
	scaled_x = x ./ IsoGK.l[1]
	scaled_xo = xo ./ IsoGK.l[1]
	d = square_metric(scaled_x, scaled_xo)
	@. σ_square*exp(-0.5*d)
end

function IsoGaussKernel()
	l = rand(1)
	σ = rand(1)
	IsoGaussKernel(l, σ)
end

reset(IsoGK::IsoGaussKernel) = IsoGaussKernel()



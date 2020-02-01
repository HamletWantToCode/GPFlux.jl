"""
Reference:
1. Structure Discovery in Nonparametric Regression through Compositional Kernel Search
		David Duvenaud, James Robert Lloyd, Roger Grosse, Joshua B. Tenenbaum, Zoubin Ghahramani (2013)

2. https://www.cs.toronto.edu/~duvenaud/cookbook/

3. GaussianProcesses.jl: A Nonparametric Bayes package for the Julia Language
"""

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
	GaussKernel(l, σ)
end

reset(GK::ArdGaussKernel) = ArdGaussKernel(length(GK.l))


## derived kernel
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
	RQKernel(l, α, σ)
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


"""
Iso Periodic kernel
"""
struct IsoPeriodKernel{T, VT<:AbstractVector{T}}
	p::VT
	l::VT
	σ::VT
end
@functor IsoPeriodKernel
function (IsoPK::IsoPeriodKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x ./ IsoPK.p[1]
	d = norm2_metric(scaled_x)
	σ_square = IsoPK.σ[1]*IsoPK.σ[1]
	l_square = IsoPK.l[1]*IsoPK.l[1]
	(@. σ_square*exp(-2.0*sinpi(d)*sinpi(d)/l_square)) + Diagonal(λ*ones(n))
end
function (IsoPK::IsoPeriodKernel{T, VT})(x, xo) where {T, VT}
	scaled_x = x ./ IsoPK.p[1]
	scaled_xo = xo ./ IsoPK.p[1]
	d = norm2_metric(scaled_x, scaled_xo)
	σ_square = IsoPK.σ[1]*IsoPK.σ[1]
	l_square = IsoPK.l[1]*IsoPK.l[1]
	@. σ_square*exp(-2.0*sinpi(d)*sinpi(d)/l_square)
end

function IsoPeriodKernel()
	p = 1.0 ./ rand(1)
	l = rand(1)
	σ = rand(1)
	IsoPeriodKernel(p, l, σ)
end

reset(IsoPK::IsoPeriodKernel) = IsoPeriodKernel()


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
	length(LK.c) == size(x, 1) && length(Lk.σv) == size(x, 1) || throw(DimensionMismatch("size of offset parameter should be the same as number of features"))
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
	LinearKernel(σb, σv, c)
end

reset(LK::ArdLinearKernel) = ArdLinearKernel(length(LK.c))



"""
White noise kernel
"""
struct WhiteNoiseKernel{T, VT<:AbstractVector{T}}
	σ::VT
end
@functor WhiteNoiseKernel
function (WNK::WhiteNoiseKernel{T, VT})(x) where {T, VT}
	n = size(x, 2)
	σ_square = WNK.σ[1]*WNK.σ[1]
	Diagonal(σ_square*ones(n))
end

function WhiteNoiseKernel()
	σ = rand(1)
	WhiteNoiseKernel(σ)
end

reset(WNK::WhiteNoiseKernel) = WhiteNoiseKernel()


"""
Constant Kernel (suffers PSD problem !!!)
"""
struct ConstantKernel{T, VT<:AbstractVector{T}}
	σ::VT
end
@functor ConstantKernel
function (CK::ConstantKernel{T, VT})(x) where {T, VT}
	n = size(x, 2)
	CK.σ[1]*ones(n, n)
end

function ConstantKernel()
	σ = rand(1)
	ConstantKernel(σ)
end

reset(CK::ConstantKernel) = ConstantKernel()


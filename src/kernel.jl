"""
Reference:
 Shengyang Sun, Guodong Zhang, Chaoqi Wang, Wenyuan Zeng, Jiaman Li, Roger Grosse (2018)
 Differentiable Compositional Kernel Learning for Gaussian Processes
"""

"""
RBF Kernel
"""
struct GaussKernel{T, VT<:AbstractVector{T}}
	l::VT
	σ::VT
end
@functor GaussKernel
function (GK::GaussKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(GK.l) == size(x, 1) || throw(DimensionMismatch("size of length scale parameter should be the same as number of features"))
	n = size(x, 2)
	scaled_x = x ./ GK.l 
	d = square_metric(scaled_x)
	σ_square = GK.σ[1]*GK.σ[1]
	(@. σ_square*exp(-0.5*d)) + Diagonal(λ*ones(n))
end

function GaussKernel(n_features::Int)
	l = rand(n_features)
	σ = rand(1)
	GaussKernel(l, σ)
end

reset(GK::GaussKernel) = GaussKernel(length(GK.l))


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

function IsoGaussKernel()
	l = rand(1)
	σ = rand(1)
	IsoGaussKernel(l, σ)
end

reset(IsoGK::IsoGaussKernel) = IsoGaussKernel()


"""
Rational quardratic kernel

Reference: https://www.cs.toronto.edu/~duvenaud/cookbook/
"""
struct RQKernel{T, VT<:AbstractVector{T}}
	l::VT
	α::VT
	σ::VT
end
@functor RQKernel
function (RQK::RQKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(RQK.l) == size(x, 1) || throw(DimensionMismatch("size of length scale parameter should be the same as number of features"))
	n = size(x, 2)
	scaled_x = x ./ RQK.l
	d = square_metric(scaled_x)
	σ_square = RQK.σ[1]*RQK.σ[1]
	(@. σ_square*exp((-RQK.α[1])*log(1 + 0.5*d/RQK.α[1]))) + Diagonal(λ*ones(n))
end

function RQKernel(n_features::Int)
	l = rand(n_features)
	α = rand(1)
	σ = rand(1)
	RQKernel(l, α, σ)
end

reset(RQK::RQKernel) = RQKernel(length(RQK.l))

## derived kernel
struct IsoRQKernel{T, VT<:AbstractVector{T}}
	l::VT
	α::VT
	σ::VT
end
@functor IsoRQKernel
function (IsoRQK::IsoRQKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x ./ IsoRQK.l
	d = square_metric(scaled_x)
	σ_square = IsoRQK.σ[1]*IsoRQK.σ[1]
	(@. σ_square*exp((-IsoRQK.α[1])*log(1 + 0.5*d/IsoRQK.α[1]))) + Diagonal(λ*ones(n))
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

function IsoPeriodKernel()
	p = 1.0 ./ rand(1)
	l = rand(1)
	σ = rand(1)
	IsoPeriodKernel(p, l, σ)
end

reset(IsoPK::IsoPeriodKernel) = IsoPeriodKernel()


"""
Linear Kernel

Reference: https://www.cs.toronto.edu/~duvenaud/cookbook/
"""
struct LinearKernel{T, VT<:AbstractVector{T}}
	σb::VT
	σv::VT
	c::VT
end
@functor LinearKernel
function (LK::LinearKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	length(LK.c) == size(x, 1) || throw(DimensionMismatch("size of offset parameter should be the same as number of features"))
	n = size(x, 2)
	shifted_x = x .- LK.c
	d = inner_prod_metric(shifted_x)
	σv_square = LK.σv[1]*LK.σv[1]
	σb_square = LK.σb[1]*LK.σb[1]
	(@. σv_square*d+σb_square) + Diagonal(λ*ones(n))
end

function LinearKernel(n_features::Int)
	σb = rand(1)
	σv = rand(1)
	c  = rand(n_features)
	LinearKernel(σb, σv, c)
end

reset(LK::LinearKernel) = LinearKernel(length(LK.c))


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


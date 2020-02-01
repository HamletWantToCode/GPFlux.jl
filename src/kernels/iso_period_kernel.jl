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



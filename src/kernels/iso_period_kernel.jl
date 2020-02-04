"""
Iso Periodic kernel
"""
struct IsoPeriodKernel{T, VT<:AbstractVector{T}} <: AbstractKernel
	lp::VT
	ll::VT
	lσ::VT
end
@functor IsoPeriodKernel
function (IsoPK::IsoPeriodKernel{T, VT})(x; λ=T(1e-6)) where {T, VT}
	n = size(x, 2)
	scaled_x = x .* exp(-IsoPK.lp[1])
	d = norm2_metric(scaled_x)
	σ_square = exp(2*IsoPK.lσ[1])
	(@. σ_square*exp(-2.0*sinpi(d)*sinpi(d)*exp(-2*IsoPK.ll[1]))) + Diagonal(λ*ones(n))
end
function (IsoPK::IsoPeriodKernel{T, VT})(x, xo) where {T, VT}
	scaled_x = x .* exp(-IsoPK.lp[1])
	scaled_xo = xo .* exp(-IsoPK.lp[1])
	d = norm2_metric(scaled_x, scaled_xo)
	σ_square = exp(2*IsoPK.lσ[1])
	@. σ_square*exp(-2.0*sinpi(d)*sinpi(d)*exp(-2*IsoPK.ll[1]))
end

function IsoPeriodKernel()
	lp = 1.0 ./ rand(1)
	ll = rand(1)
	lσ = rand(1)
	IsoPeriodKernel(lp, ll, lσ)
end

reset(IsoPK::IsoPeriodKernel) = IsoPeriodKernel()



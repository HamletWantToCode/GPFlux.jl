import Base: eltype, length
import Distributions: _rand!, _logpdf, show_multline
using Distributions: ContinuousMultivariateDistribution
using Random: AbstractRNG



function chol_tril(X)
	C = cholesky(X)
	C.L
end

"""
MvNormal(μ::AbstractVector, Σ::AbstractMatrix) -> customized MvNormal distribution

the `MvNormal` provided by Distributions.jl makes it complicated to implement a Zygote @adjoint` 
"""
struct MvNormal{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}} <: ContinuousMultivariateDistribution
	μ::VT
	Σ::MT
end

function MvNormal(gp::GaussProcess, x::AbstractArray)
	n = size(x, 2)
	μ = reshape(gp.mean(x), n)
	K = gp.kernel(x) + Diagonal(exp(2*gp.lnoise[1])*ones(n))
	MvNormal(μ, K)
end


eltype(::Type{<:MvNormal{T}}) where {T} = T
length(d::MvNormal) = length(d.μ)

function _rand!(rng::AbstractRNG, d::MvNormal, x::VecOrMat)
	ϵ = randn(rng, size(x))
	L = chol_tril(d.Σ)
	y = d.μ .+ L*ϵ
	y
end

function _logpdf(d::MvNormal, x::AbstractVector)
	y = x - d.μ
	L = chol_tril(d.Σ)
	z = L \ y
	d = length(d)
	pdf = -0.5*dot(z, z) - 0.5*d*log(2π) - logdet(L)
	pdf
end

Base.show(io::IO, d::MvNormal) =
    show_multline(io, d, [(:dim, length(d)), (:μ, d.μ), (:Σ, d.Σ)])


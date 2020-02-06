"""
Gauss Process
"""
struct GaussProcess{MT, KT, VT}
	mean::MT
	kernel::KT
	lnoise::VT
end
@functor GaussProcess

function Base.show(io::IO, gp::GaussProcess)
	print(io, "GaussProcess(", gp.mean, ", ", gp.kernel, ", ", "lnoise=", gp.lnoise[1], ")")
end


"""
Construct multivariate normal distribution from GP, facilitate sampling
"""
function MvNormal(gp::GaussProcess, x::AbstractArray)
	n = size(x, 2)
	μ = reshape(gp.mean(x), n)
	K = gp.kernel(x) + Diagonal(exp(2*gp.lnoise[1])*ones(n))
	MvNormal(μ, K)
end


"""
loss function, negative log likelihood
"""
# loss function
function negloglik(gp::GaussProcess, x, y; λ=1e-6)
	μ = reshape(gp.mean(x), size(y))
	Σ = gp.kernel(x, λ=λ) + Diagonal(exp(2*gp.lnoise[1])*ones(size(x, 2)))

  d = length(y)
  # L = BackwardsLinalg.cholesky(Σ)
	C = cholesky(Σ)
	L = C.L
	ȳ = y .- μ
	z = L \ ȳ
  0.5*dot(z, z) + (d/2.0)*log(2π) + logdet(L)
end


"""
inference from fitted GP
"""
# inference
function predict(gp::GaussProcess, x, x_old, y_old; λ=1e-6)
	μn = ndims(y_old)>1 ? gp.mean(x) : reshape(gp.mean(x), size(x, 2))
	μo = reshape(gp.mean(x_old), size(y_old))
	Σ_oo = gp.kernel(x_old, λ=λ) + Diagonal(exp(2*gp.lnoise[1])*ones(size(x_old, 2)))
	Σ_no = gp.kernel(x, x_old)
	Σ_nn = gp.kernel(x, λ=λ)

	L = BackwardsLinalg.cholesky(Σ_oo)
	ȳo = y_old .- μo
	zo = L \ ȳo
	α = L' \ zo
	k = L \ Σ_no'
	k̄ = L' \ k

	yn = μn .+ reshape(Σ_no*α, size(μn))
	Σ̄_nn = Σ_nn .- Σ_no*k̄
	
	yn, diag(Σ̄_nn)
end


"""
dispatch parameters (Array) to a ML model, facilitate use of Optim.jl
"""
function nparams(ps)
	nps = []
	for p in ps
		push!(nps, length(p))
	end
	nps
end

function flatten_params(ps)
	psarray = []
	for p in ps
		push!(psarray, vec(p))
	end
	vcat(psarray...)
end

function dispatch!(model, x::AbstractVector)
	ps = params(model)
	nps = nparams(ps)

	loc = 1
	i = 1
	for p in ps
		p .= reshape(x[loc:loc+nps[i]-1], size(p))
		loc += nps[i]
		i += 1
	end

	model
end




using LinearAlgebra
using GPFlux
using Zygote
using Random
using GaussianProcesses
using Test


@testset "negative logliklihood" begin
	Random.seed!(4)
	d = 4
	n = 10
	X = rand(d, n)
	y = rand(n)

	# GaussianProcesses GP
	gp_zmean = MeanZero()
	gp_se_kernel = SE(4.0, 3.0)
	gp_lognoise = -1.0
	gp_gp = GP(X, y, gp_zmean, gp_se_kernel, gp_lognoise)
	PT = GaussianProcesses.FullCovariancePrecompute(n)
	CST = GaussianProcesses.FullCovariance()
	GaussianProcesses.update_mll_and_dmll!(gp_gp, PT)
	gp_nll = -gp_gp.mll
	gp_grads = -gp_gp.dmll

	# GPFlux GP
	zmean = ConstantMean()
	se_kernel = IsoGaussKernel([4.0], [3.0])
	noise2 = [-1.0]
	gp = GaussProcess(zmean, se_kernel, noise2)
	ps = params(gp)
	my_nll = negloglik(gp, X, y)
	gs = gradient(()->negloglik(gp, X, y), ps)
	my_grads = [gs[p][1] for p in ps if p!=[0.0]]
	
	@show gp_nll
	@show my_nll
	@show sort(gp_grads)
	@show sort(my_grads)

	@test isapprox(my_nll, gp_nll, rtol=1e-3, atol=1e-8)
	@test isapprox(sort(gp_grads), sort(my_grads), rtol=1e-4, atol=1e-8)
end




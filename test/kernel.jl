using DiffGP
using LinearAlgebra
using Test, Random


@testset "kernel_psd" begin
	Random.seed!(4)
	T = Float64
	M = [1, 4, 10]
	n = 10

	for m in M
		X = rand(T, m, n)
		@show size(X)

		rbf_kernel = GaussKernel(m)
		isorbf_kernel = IsoGaussKernel()
		rq_kernel = RQKernel(m)
		isorq_kernel = IsoRQKernel()
		isoperiod_kernel = IsoPeriodKernel()
		linear_kernel = LinearKernel(m)
		# wn_kernel = WhiteNoiseKernel()
 		# c_kernel = ConstantKernel()
		kernels = [rbf_kernel, isorbf_kernel, rq_kernel, isorq_kernel, isoperiod_kernel, linear_kernel, wn_kernel]

		@show isposdef(rbf_kernel(X))
		@show isposdef(isorbf_kernel(X))
    @show isposdef(rq_kernel(X))
		@show isposdef(isorq_kernel(X))
		@show isposdef(isoperiod_kernel(X))
		@show isposdef(linear_kernel(X))
		# @show isposdef(wn_kernel(X))
		# @show isposdef(c_kernel(X))
		@test all([isposdef(K(X)) for K in kernels])
	end
end


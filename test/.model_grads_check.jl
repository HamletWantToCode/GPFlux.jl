using GPFlux
using Test, Random
using Flux


@testset "model_gradient_check" begin
	Random.seed!(4)
	T = Float64
	X = rand(T, 4, 10)
	y = rand(T, 10)
	
	# kernels
	rbf_kernel = GaussKernel(4)
	isorbf_kernel = IsoGaussKernel()
	rq_kernel = RQKernel(4)
	isorq_kernel = IsoRQKernel()
	isoperiod_kernel = IsoPeriodKernel()
	linear_kernel = LinearKernel(4)
	kernels = [rbf_kernel, isorbf_kernel, rq_kernel, isorq_kernel, isoperiod_kernel, linear_kernel]

	# means
	c_mean = ConstantMean()
	simple_nn_mean = SimpleNeuralNetworkMean(4, 1)
	means = [c_mean, simple_nn_mean]

	for kernel in kernels, mean in means
		mean = reset(mean)
		kernel = reset(kernel)
		@show (mean, kernel)
		gp = GaussProcess(mean, kernel)
		ps = params(gp)
		f = () -> negloglik(gp, X, y)
		@test model_gradient_check(f, ps)
	end
end

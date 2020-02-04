using GPFlux
using LinearAlgebra
using GaussianProcesses
using Statistics
using Calculus
using Zygote
using Test, Random


function K_check(mykernel, gpkernel, X)
  myK = mykernel(X)
	gpK = cov(gpkernel, X) + Diagonal(1e-6*ones(size(X, 2)))
	@show isposdef(myK)
	@show isposdef(gpK)
	isposdef(myK) == isposdef(gpK) &&  isapprox(myK, gpK, rtol=1e-3, atol=1e-8) 
end

function dKdθ_check(mykernel, gpkernel, X, v)
  N = size(X, 2)
  
  myps = GPFlux.params(mykernel)
  f = () -> v'*mykernel(X)*v
  mygs = gradient(f, myps)
  mygv = [mygs.grads[p][1] for p in myps]
  
  n_gpps = GaussianProcesses.num_params(gpkernel)
  gpps = GaussianProcesses.get_params(gpkernel)
  gpdKdθ = zeros(N, N, n_gpps)
  gpmetric = GaussianProcesses.KernelData(gpkernel, X, X)
  GaussianProcesses.grad_stack!(gpdKdθ, gpkernel, X, X, gpmetric)
  gpgv = [tr(v*v'*gpdKdθ[:, :, i]) for i in 1:n_gpps]

	# gpgv = [tr(v*v'*gpdKdθ[:, :, i]).*exp(-gpps[i]) for i in 1:n_gpps]
  
#  calps = [p[1] for p in myps]
#  calgv = Calculus.gradient(calps) do ps
#      ps_ = [[p] for p in ps]
#      kernel = typeof(mykernel)(ps_...)
#      v'*kernel(X)*v
#  end
  
	@show sort(mygv)
# 	@show sort(calgv)
	@show sort(gpgv)
#  isapprox(sort(mygv),sort(calgv),rtol=1e-3,atol=1e-8) && 
	isapprox(sort(mygv),sort(gpgv),rtol=1e-3,atol=1e-8)
end


@testset "kernel check" begin
	rng = MersenneTwister(4)
	d = 4
	n = 10
	X = rand(rng, d, n)
	v = rand(rng, n)
	kernel_pairs = [(IsoGaussKernel([4.0], [1.0]), SE(4.0, 1.0)),
									(IsoPeriodKernel([0.0], [0.0], [1.0]), Periodic(0.0, 1.0, 0.0)),
									(IsoRQKernel([0.0], [-1.0], [0.0]), RQ(0.0, 0.0, -1.0)),
									(IsoLinearKernel([0.0]), Lin(0.0)),
								  (SE_add_PeriodKernel(IsoGaussKernel([4.0],[0.0]),IsoPeriodKernel([0.0],[0.0],[1.0])), SE(4.0,0.0)+Periodic(0.0,1.0,0.0)),
									(SE_mul_PeriodKernel(IsoGaussKernel([4.0],[0.0]),IsoPeriodKernel([0.0],[0.0],[1.0])), SE(4.0,0.0)*Periodic(0.0,1.0,0.0)),
									]

	for (mykernel, gpkernel) in kernel_pairs
		@show (mykernel, gpkernel)
		@test K_check(mykernel, gpkernel, X)
		@test dKdθ_check(mykernel, gpkernel, X, v)
	end

end


	





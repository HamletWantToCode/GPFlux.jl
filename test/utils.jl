using GPFlux
using Zygote
using Calculus
using Test, Random


@testset "power function (scalar)" begin
	Random.seed!(4)
	xs = 10*rand(4)
	αs = [-0.5, 3.5]

	for x in xs, α in αs
		dfdx, dfdα = gradient(GPFlux.power, x, α)
		Finit_dfdx, Finit_dfdα = Calculus.gradient(z->GPFlux.power(z[1], z[2]), [x, α])
		@test dfdx ≈ Finit_dfdx
		@test dfdα ≈ Finit_dfdα
	end
end

@testset "power function (Array)" begin
	Random.seed!(4)
	x = rand(3, 3)
	w = rand(3)
	f(x, α) = w'*GPFlux.power(x, α)*w
	
	dfdx, dfdα = gradient(f, x, 2.5)
	Finit_dfdα = Calculus.gradient(z->f(x, first(z)), [2.5])
	@test dfdα ≈ Finit_dfdα[1]
end


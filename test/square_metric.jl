using DiffGP
using LinearAlgebra
using Test, Random


@testset "square_metric" begin
  Random.seed!(3)
  T = Float64
	for N in [2, 4, 7]
    @show N
    X = rand(T, 4, N)
		v = rand(T, N)

    function tfunc(x)
			D = square_metric(x)
			(v'*Symmetric(D, :L)*v)[] |> real
    end
    @test gradient_check(tfunc, X)
  end
end



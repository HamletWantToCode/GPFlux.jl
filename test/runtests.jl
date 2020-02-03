using Test


@testset "metric" begin
	include("metric.jl")
end

@testset "kernel" begin
	include("kernel.jl")
end

@testset "negloglik" begin
	include("negloglik.jl")
end

@testset "utils" begin
	include("utils.jl")
end


abstract type AbstractKernel end

include("gauss_kernel.jl")
include("rq_kernel.jl")
include("iso_period_kernel.jl")
include("linear_kernel.jl")
include("neural_kernel_network.jl")
include("composite_kernel.jl")


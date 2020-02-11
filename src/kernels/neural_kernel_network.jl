"""
Reference:
 Shengyang Sun, Guodong Zhang, Chaoqi Wang, Wenyuan Zeng, Jiaman Li, Roger Grosse (2018)
 Differentiable Compositional Kernel Learning for Gaussian Processes
"""

using Base: tail
using Flux: glorot_uniform
import Flux: functor


"""
Neural Kernel Network
modify https://github.com/FluxML/Flux.jl/tree/master/src/layers/basic.jl Chain
"""
struct NeuralKernelNetwork{T<:Tuple} <: AbstractKernel
	layers::T
	NeuralKernelNetwork{T}(ls...) where {T} = new{T}(ls)
end
NeuralKernelNetwork(ls...) = NeuralKernelNetwork{typeof(ls)}(ls...)

functor(nkn::NeuralKernelNetwork) = nkn.layers, ls->NeuralKernelNetwork(ls...)

applynkn(::Tuple{}, x) = x
applynkn(fs::Tuple, x) = applynkn(tail(fs), first(fs)(x))
applynkn(fs::Tuple, x, xo) = applynkn(tail(fs), first(fs)(x, xo))

function (nkn::NeuralKernelNetwork)(x; λ=1e-6)
	N = size(x, 2)
	x1 = applynkn(nkn.layers, x)
	K = reshape(x1, N, N)
	K + Diagonal(λ*ones(N))
end

function (nkn::NeuralKernelNetwork)(x, xo)
	M = size(x, 2)
	N = size(xo, 2)
	x1 = applynkn(nkn.layers, x, xo)
	K = reshape(x1, M, N)
	K
end

function Base.show(io::IO, kernel::NeuralKernelNetwork)
	print(io, "NeuralKernelNetwork(")
	join(io, kernel.layers, ", ")
	print(io, ")")
end


"""
Primitive layer, constituted from basic kernels
"""
struct Primitive{T<:Tuple{Vararg{<:AbstractKernel}}}
  kernels::T
  Primitive(ks...) = new{typeof(ks)}(ks)
end
functor(p::Primitive) = p.kernels, ks -> Primitive(ks...)

function (p::Primitive)(x)
	Ks = [reshape(Ker(x), 1, :) for Ker in p.kernels]
	vcat(Ks...)
end
function (p::Primitive)(x, xo)
	Ks = [reshape(Ker(x, xo), 1, :) for Ker in p.kernels]
	vcat(Ks...)
end

function Base.show(io::IO, layer::Primitive)
	print(io, "Primitive(")
	join(io, layer.kernels, ", ")
	print(io, ")")
end


"""
Linear layer: weigths & bias should be positive
"""
positive(x) = @. log(1.0 + exp(x)) + 1e-6  # plus 1e-6 to avoid this value to be 0.0
positive_back(x, dy) = @. dy*(1.0/(1.0+exp(-x))) 

struct Linear{WT, BT}
	W::WT
	b::BT
end
@functor Linear
function (lin::Linear)(x)
	pos_W = positive(lin.W) 
	pos_b = positive(lin.b)
	pos_W*x .+ pos_b
end
function Linear(in::Integer, out::Integer; init_W=glorot_uniform, init_b=zeros)
	Linear(init_W(out, in), init_b(out))
end

function Base.show(io::IO, layer::Linear)
	print(io, "Linear(", size(layer.W, 2), ",", size(layer.W, 1), ")")
end


"""
Product layer
"""
function Product(x; step=2)
	m, n = size(x)
	m%step == 0 || error("the first dimension of inputs must be multiple of step")
	new_x = reshape(x, step, m÷step, n)
	.*([new_x[i, :, :] for i in 1:step]...)
end


"""
common used all product and all sum
"""
# not use `prod` here, since it'll results in unknown overflow in backpropogation
allProduct(x) = .*([x[i,:] for i in 1:size(x, 1)]...)

# similar reason, not use `sum`
function allSum(x)
	m = size(x, 1)
	v = ones(eltype(x), m)
	v'*x
end

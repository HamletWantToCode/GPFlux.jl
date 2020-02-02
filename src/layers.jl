"""
Reference:
 Shengyang Sun, Guodong Zhang, Chaoqi Wang, Wenyuan Zeng, Jiaman Li, Roger Grosse (2018)
 Differentiable Compositional Kernel Learning for Gaussian Processes
"""

using Base: tail
using Flux: glorot_uniform
import Flux: functor, Chain, applychain


"""
Primitive layer, constituted by basic kernels
"""
struct Primitive{T<:Tuple}
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


"""
extend Chain
"""
applychain(fs::Tuple, x, xo) = applychain(tail(fs), first(fs)(x, xo))
(c::Chain)(x, xo) = applychain(c.layers, x, xo)


"""
Linear layer: weigths & bias should be positive
"""
positive(x) = @. log(1.0 + exp(x))
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


"""
Product layer
"""
function Product(x; step=2)
	m, n = size(x)
	m%step == 0 || error("the first dimension of inputs must be multiple of step")
	new_x = reshape(x, step, m÷step, n)
	.*([x[i, :, :] for i in 1:step]...)
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

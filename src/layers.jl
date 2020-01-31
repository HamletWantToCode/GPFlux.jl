"""
Reference:
 Shengyang Sun, Guodong Zhang, Chaoqi Wang, Wenyuan Zeng, Jiaman Li, Roger Grosse (2018)
 Differentiable Compositional Kernel Learning for Gaussian Processes
"""

using Flux: glorot_uniform


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
	new_x = reshape(x, m÷step, step, n)
	dropdims(prod(new_x, dims=1), dims=2)
end


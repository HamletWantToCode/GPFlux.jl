"""
Constant mean
"""
struct ConstantMean{T}
	c::AbstractVector{T}
end
@functor ConstantMean
(cm::ConstantMean{T})(x) where {T} = zeros(T, size(x, 2)) .+ cm.c

function ConstantMean()
	ConstantMean([0.0])
end

reset(CM::ConstantMean) = ConstantMean()


"""
simple neural network mean (single layer, linear)
"""
SimpleNeuralNetworkMean(n_features, n_out) = Dense(n_features, n_out)

reset(SNNM::Dense) = Dense(size(SNNM.W, 2), size(SNNM.W, 1))


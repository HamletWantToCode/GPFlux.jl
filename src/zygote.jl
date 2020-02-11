using Zygote: @adjoint, pullback
using Flux: Optimise


@adjoint function square_metric(x)
	square_metric(x), dy -> (square_metric_back(x, dy),)
end

@adjoint function norm2_metric(x)
	D = norm2_metric(x)
	D, dy -> (norm2_metric_back(x, D, dy),)
end

@adjoint function inner_prod_metric(x)
	inner_prod_metric(x), dy -> (inner_prod_metric_back(x, dy),)
end

@adjoint function positive(x)
	positive(x), dy -> (positive_back(x, dy),)
end

power(x, α) = x.^α
@adjoint function power(x, α)
	y = power(x, α)
	y, function (dy)
		α*(@. dy*x^(α-1)), tr(dy'* (@. y*log(x)))
	end
end

@adjoint function rand(mvn::MvNormal, N::Int)
  ϵ = randn(length(mvn), N)
  L, chol_back = pullback(chol_tril, mvn.Σ)
  y = mvn.μ .+ L*ϵ
  y, function (ȳ)
    μ̄ = sum(ȳ, dims=2)
    L̄ = zero(L)
    for i in 1:N
      L̄ += ȳ[:,i] * ϵ[:,i]'
    end
    Σ̄_half, = chol_back(L̄)
    Σ̄ = 0.5*(Σ̄_half + Σ̄_half')
    (μ=μ̄, Σ=Σ̄), nothing
  end
end



# gradient check for functions
function gradient_check(f, args...; η = 1e-5)
  g = gradient(f, args...)
  dy_expect = η*sum(abs2.(g[1]))
  dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
  @show dy
  @show dy_expect
  isapprox(dy, dy_expect, rtol=1e-3, atol=1e-8)
end

# gradient check for models
function model_gradient_check(f, ps; η = 1e-5)
    g = gradient(f, ps)
    dy_expect = η*sum([gp == nothing ? 0.0 : sum(abs2.(gp)) for gp in values(g.grads)])

    f_original = f()
    for p in ps
			g[p] == nothing ? continue : Optimise.update!(p, -η.*g[p])
    end
    f_new = f()
    dy = f_original - f_new
  @show dy
  @show dy_expect
  isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

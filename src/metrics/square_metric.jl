"""
(D)_{ij} = ∑_k (x_k^{(i)} - x_k^{(j)})^2
"""

function square_metric(x)
	N = size(x, 2)
	D = zeros(eltype(x), N, N)
	@inbounds for i in 1:N
		for j in i+1:N
			dx = norm(view(x, :, i) .- view(x, :, j))
			D[i,j] = D[j,i] = dx * dx
		end
	end
	D
end

function square_metric_back(x, sym_dD)
	dD = 0.5*(sym_dD + sym_dD')  # Cholesky decomposition only use lower triangle, reconstruct the true dD
	m, n = size(x)
	dx = zeros(eltype(x), m, n)
  @inbounds for i in 1:m
    for j in 1:n
      for k in 1:n
        dx[i,j] += 4*dD[j,k]*(x[i,j]-x[i,k])
      end
    end
  end
  dx
end

function square_metric(x, xo)
	N1 = size(x, 2)
	N2 = size(xo, 2)
	D = zeros(eltype(x), N1, N2)
	@inbounds for i in 1:N1
		for j in 1:N2
			dx = norm(view(x, :, i) .- view(xo, :, j))
			D[i,j] = dx * dx
		end
	end
	D
end


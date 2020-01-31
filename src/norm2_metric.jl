function norm2_metric(x)
	m, n = size(x)
	D = zeros(eltype(x), n, n)
	@inbounds for i in 1:n
		for j in i+1:n
			D[i,j] = D[j,i] = norm(view(x, :, i) .- view(x, :, j))
		end
	end
	D
end

function norm2_metric_back(x, D, sym_dD)
	dD = 0.5*(sym_dD + sym_dD')  # Cholesky decomposition only use lower triangle, reconstruct the true dD
	m, n = size(x)
	T = eltype(x)
	dx = zeros(T, m, n)
	@inbounds for i in 1:m
		for j in 1:n
			for k in 1:n
				dx[i,j] += 2.0*(dD[j,k]/(D[j,k]+eps(T)))*(x[i,j]-x[i,k])
			end
		end
	end
	dx
end


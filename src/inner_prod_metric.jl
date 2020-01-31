function inner_prod_metric(x)
	n = size(x, 2)
	D = zeros(eltype(x), n, n)
	@inbounds for i in 1:n
		D[i,i] = dot(view(x, :, i), view(x, :, i))
		for j in i+1:n
			D[i,j] = D[j,i] = dot(view(x, :, i), view(x, :, j))
		end
	end
	D
end

function inner_prod_metric_back(x, sym_dD)
	dD = 0.5*(sym_dD + sym_dD')
	m, n = size(x)
	dx = zeros(eltype(x), m, n)
	@inbounds for i in 1:m
		for j in 1:n
			for k in 1:n
				dx[i,j] += 2*dD[j,k]*x[i,k]
			end
		end
	end
	dx
end


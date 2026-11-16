# Test problems, taken from the test suite of Krylov.jl (`test/test_utils.jl`,
# `test/get_div_grad.jl`).

# Identity matrix.
eye(n::Int; FC=Float64) = sparse(one(FC) * I, n, n)

# Based on Lars Ruthotto's initial implementation.
function get_div_grad(n1 :: Int, n2 :: Int, n3 :: Int)

  # Divergence
  D1 = kron(eye(n3), kron(eye(n2), ddx(n1)))
  D2 = kron(eye(n3), kron(ddx(n2), eye(n1)))
  D3 = kron(ddx(n3), kron(eye(n2), eye(n1)))

  # DIV from faces to cell-centers
  Div = [D1 D2 D3]

  return Div * Div'
end

# 1D finite difference on staggered grid
function ddx(n :: Int)
  e = ones(n)
  return sparse([1:n; 1:n], [1:n; 2:n+1], [-e; e])
end

# Symmetric and positive definite systems.
function symmetric_definite(n :: Int=10; FC=Float64)
  α = FC <: Complex ? FC(im) : one(FC)
  A = spdiagm(-1 => α * ones(FC, n-1), 0 => 4 * ones(FC, n), 1 => conj(α) * ones(FC, n-1))
  b = A * FC[1:n;]
  return A, b
end

# Sparse Laplacian.
function sparse_laplacian(n :: Int=16; FC=Float64)
  A = get_div_grad(n, n, n)
  b = ones(n^3)
  return A, b
end


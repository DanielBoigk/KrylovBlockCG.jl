# KrylovBlockCG.jl

Block conjugate gradients for [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl):
Hermitian positive (semi)definite systems `AX = B` with several right-hand sides, with an
optional projection onto the complement of a known null space of `A`.

The package works with released Krylov.jl (0.10.12 and later 0.10 versions). It adds
`block_cg`, `block_cg!` and the workspace `BlockCgWorkspace`, a `Krylov.BlockKrylovWorkspace`.
Krylov's generic interface (`krylov_workspace`, `krylov_solve`, `krylov_solve!` with the method
`:block_cg`) and its workspace accessors (`Krylov.solution`, `Krylov.statistics`, `Krylov.iteration_count`, ...) work
with it.

## Installation

The package is not in the General registry.

```julia
using Pkg
Pkg.add(url = "https://github.com/DanielBoigk/KrylovBlockCG.jl")
```

## Use

```julia
using Krylov, KrylovBlockCG, SparseArrays, LinearAlgebra, Printf

# Weighted graph Laplacian of an m × m grid: a pure Neumann problem, positive semidefinite
# with the constants as null space.
m = 40
n = m^2
idx(i, j) = (j - 1) * m + i
I, J, V = Int[], Int[], Float64[]
for j in 1:m, i in 1:m, (di, dj) in ((1, 0), (0, 1))
    (i + di > m || j + dj > m) && continue
    a, b = idx(i, j), idx(i + di, j + dj)
    w = 1 + 9 * (sin(a) + 1) / 2                 # conductivities between 1 and 10
    append!(I, (a, b, a, b)); append!(J, (a, b, b, a)); append!(V, (w, w, -w, -w))
end
A = sparse(I, J, V, n, n)

# Four right-hand sides: sources on the boundary, with zero sum (compatible data)
B = zeros(n, 4)
for (k, (a, b)) in enumerate(((1, n), (m, n - m + 1), (idx(1, m ÷ 2), idx(m, m ÷ 2)), (idx(m ÷ 2, 1), idx(m ÷ 2, m))))
    B[a, k], B[b, k] = 1.0, -1.0
end

# Solve AX = B on the range of A, with the solution orthogonal to the constants
X, stats = block_cg(A, B; nullspace = ones(n))
@printf("Relative residual: %8.1e\n", norm(A * X - B) / norm(B))

# The same with a Jacobi preconditioner (M ≈ A⁻¹, applied with mul!)
X, stats = block_cg(A, B; nullspace = ones(n), M = inv(Diagonal(diag(A))))

# In place, reusing the memory across solves
workspace = BlockCgWorkspace(A, B; nullspace = ones(n))
block_cg!(workspace, A, B)
X = Krylov.solution(workspace)
```

`?block_cg` documents the keyword arguments.

## Method

The block iteration is the one of O'Leary (1980), in Galerkin form. Every search block is
orthonormalized by a rank-revealing eigendecomposition of its Gram matrix, so (nearly) linearly
dependent right-hand sides and search directions are dropped instead of causing a breakdown, and
converged columns are removed from the block.

For a singular `A` with a known null space `V` (for example the stiffness matrix of a pure
Neumann problem, `V` = constants), every residual and every preconditioned residual is projected
onto the range of `A`, so the iteration never leaves it. The component of the solution in `V` is
fixed at the end by `Wᴴx = 0`, where `W` is the keyword `grounding` (default: `V` itself, which
gives the solution orthogonal to the null space).

- D. P. O'Leary, The block conjugate gradient algorithm and related methods, Linear Algebra and
  its Applications 29 (1980) 293–322, doi:10.1016/0024-3795(80)90247-5

## Origin

The solver was written as a branch of a Krylov.jl fork
([DanielBoigk/Krylov.jl](https://github.com/DanielBoigk/Krylov.jl), `block-cg`) and moved into a
package of its own so that it can be used with released Krylov.jl. It is written with internal
functions of Krylov.jl, and it defines the method `:block_cg` of Krylov's generic interface from
outside Krylov.jl; both can break with a new Krylov.jl release, which is why the supported
versions are bounded.

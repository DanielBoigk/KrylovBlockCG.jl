"""
    KrylovBlockCG

Block conjugate gradients for [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl):
[`block_cg`](@ref) and [`block_cg!`](@ref) with the workspace [`BlockCgWorkspace`](@ref), for
Hermitian positive (semi)definite systems `AX = B` with several right-hand sides, with an
optional projection onto the complement of a known null space.

The solver is a block Krylov method in the sense of Krylov.jl: `BlockCgWorkspace` is a
`Krylov.BlockKrylovWorkspace`, and Krylov's generic interface (`krylov_workspace`,
`krylov_solve`, `krylov_solve!` with the method `:block_cg`) and its workspace accessors
(`solution`, `statistics`, `iteration_count`, ...) work with it.
"""
module KrylovBlockCG

using LinearAlgebra, Printf
using Krylov

# Internal names of Krylov.jl that the solver is written with.
using Krylov: FloatOrComplex, @kswap!, allocate_if, extract_parameters, kdisplay, ktimer, ktypeof,
              matrix_to_vector, reset!

# Functions of Krylov.jl that get methods for `BlockCgWorkspace` and for the method `:block_cg`.
import Krylov: krylov_workspace, krylov_solve, krylov_solve!,
               elapsed_time, elapsed_allocation_time, statistics, solution_count, iteration_count,
               Aprod_count, Atprod_count, solution, results, issolved, warm_start!

include("workspace.jl")
include("block_cg.jl")
include("interface.jl")

end # module

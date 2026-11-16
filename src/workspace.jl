export BlockCgWorkspace


"""
Workspace for the in-place methods [`block_cg!`](@ref) and [`krylov_solve!`](@ref).

The following outer constructors can be used to initialize this workspace:

    workspace = BlockCgWorkspace(m, n, p, SV, SM; nullspace = nothing, grounding = nothing)
    workspace = BlockCgWorkspace(A, B; nullspace = nothing, grounding = nothing)

`m` and `n` denote the dimensions of the linear operator `A` passed to the in-place methods.
`p` denotes the number of columns of the right-hand side `B` passed to the in-place methods.
`SV` is the storage type of the vectors in the workspace, such as `Vector{Float64}`.
`SM` is the storage type of the matrices in the workspace, such as `Matrix{Float64}`.

For a singular `A`, `nullspace` is a basis of its null space (an `n × k` matrix or a vector,
not necessarily orthonormal), and `grounding` (`n × k` or a vector) defines the solution among
all solutions: `Wᴴx = 0` for `W = grounding`. By default `W` is the null space itself, which
gives the solution orthogonal to it. `WᴴV` must be invertible. Both are given on the host and
copied into the storage type `SM`. See [`block_cg`](@ref).

After a solve, `workspace.bnorm` and `workspace.rnorm` hold the norms of the projected
right-hand sides and the final residual norms column by column, and
`workspace.defect` the relative part `‖(I - Π)B‖_F / ‖B‖_F` of the right-hand
side outside the range of `A` that was removed.
"""
mutable struct BlockCgWorkspace{T,FC,SV,SM} <: BlockKrylovWorkspace{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  R          :: SM
  Z          :: SM
  P          :: SM
  Q          :: SM
  V          :: SM
  W          :: SM
  F          :: SM
  K          :: SM
  G          :: SM
  C          :: SM
  mask       :: SM
  colsums    :: SM
  maskh      :: Matrix{FC}
  colsumsh   :: Vector{FC}
  Gh         :: Matrix{FC}
  Ch         :: Matrix{FC}
  Th         :: Matrix{FC}
  Eh         :: Matrix{FC}
  Lh         :: Matrix{FC}
  Wh         :: Matrix{FC}
  λh         :: Vector{T}
  perm       :: Vector{Int}
  chol_ok    :: Bool
  d          :: Vector{T}
  bnorm      :: Vector{T}
  rnorm      :: Vector{T}
  tol        :: Vector{T}
  active     :: Vector{Bool}
  defect     :: T
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockCgWorkspace(m::Integer, n::Integer, p::Integer, SV::Type, SM::Type; nullspace = nothing, grounding = nothing)
  start_allocation_time = time_ns()
  FC = eltype(SV)
  T  = real(FC)
  if nullspace === nothing
    grounding === nothing || throw(ArgumentError("a grounding needs a null space basis"))
    Vh = Wh = Fh = Matrix{FC}(undef, n, 0)
  else
    Vh = Matrix{FC}(reshape(collect(nullspace), size(nullspace, 1), :))
    size(Vh, 1) == n || throw(DimensionMismatch("the null space basis must have $n rows"))
    Vh = Matrix(qr(Vh).Q)  # orthonormal basis of the null space
    k = size(Vh, 2)
    Wh = grounding === nothing ? copy(Vh) : Matrix{FC}(reshape(collect(grounding), size(grounding, 1), :))
    size(Wh) == (n, k) || throw(DimensionMismatch("the grounding must be $n × $k"))
    WV = Wh' * Vh
    abs(det(WV)) > eps(T) * opnorm(Wh) || throw(ArgumentError("Wᴴ V must be invertible"))
    Fh = Vh / WV
  end
  k = size(Vh, 2)
  ΔX = SM(undef, 0, 0)
  X  = SM(undef, n, p)
  R  = SM(undef, n, p)
  Z  = SM(undef, n, p)
  P  = SM(undef, n, p)
  Q  = SM(undef, n, p)
  V  = copyto!(SM(undef, n, k), Vh)
  W  = copyto!(SM(undef, n, k), Wh)
  F  = copyto!(SM(undef, n, k), Fh)
  K  = SM(undef, k, p)
  G  = SM(undef, p, p)
  C  = SM(undef, p, p)
  mask = SM(undef, 1, p)
  colsums = SM(undef, 1, p)
  SV = isconcretetype(SV) ? SV : matrix_to_vector(typeof(X))
  SM = isconcretetype(SM) ? SM : typeof(X)
  stats = SimpleStats(0, false, false, false, 0, T[], T[], T[], T[], 0.0, 0.0, "unknown")
  host() = zeros(FC, p, p)
  workspace = BlockCgWorkspace{T,FC,SV,SM}(m, n, p, ΔX, X, R, Z, P, Q, V, W, F, K, G, C, mask, colsums,
                                           ones(FC, 1, p), zeros(FC, p), host(), host(), host(), host(), host(), host(),
                                           zeros(T, p), collect(1:p), true, zeros(T, p), zeros(T, p), zeros(T, p),
                                           zeros(T, p), fill(true, p), zero(T), false, stats)
  workspace.stats.allocation_timer = start_allocation_time |> ktimer
  return workspace
end

function BlockCgWorkspace(A, B; nullspace = nothing, grounding = nothing)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockCgWorkspace(m, n, p, SV, SM; nullspace, grounding)
end


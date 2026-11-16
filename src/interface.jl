# Generic interface and workspace accessors for `block_cg`. This is the code that Krylov.jl
# generates for its block methods (`src/interface.jl` and `src/workspace_accessors.jl`),
# here for the one entry of `block_cg`.

for (workspace, krylov, args, def_args, optargs, def_optargs, kwargs, def_kwargs, kwargs_workspace, def_kwargs_workspace) in [
  (:BlockCgWorkspace    , :block_cg    , args_block_cg    , def_args_block_cg    , optargs_block_cg    , def_optargs_block_cg    , kwargs_block_cg    , def_kwargs_block_cg    , kwargs_workspace_block_cg   , def_kwargs_workspace_block_cg   )
]
  # Create the symbol for the in-place method
  krylov! = Symbol(krylov, :!)

  ## Generic constructors for each subtype of BlockKrylovWorkspace
  if isempty(kwargs_workspace)
      @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, p::Integer, SV::Type, SM::Type) = $workspace(m, n, p, SV, SM)
      @eval krylov_workspace(::Val{Symbol($krylov)}, A, B) = $workspace(A, B)
  else
      @eval krylov_workspace(::Val{Symbol($krylov)}, m::Integer, n::Integer, p::Integer, SV::Type, SM::Type; $(def_kwargs_workspace...)) = $workspace(m, n, p, SV, SM; $(kwargs_workspace...))
      @eval krylov_workspace(::Val{Symbol($krylov)}, A, B; $(def_kwargs_workspace...)) = $workspace(A, B; $(kwargs_workspace...))
  end

  ## Out-of-place
  if isempty(kwargs_workspace)
    @eval begin
      function $(krylov)($(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        workspace = $workspace($(args...))
        elapsed_time = ktimer(start_time)
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); $(kwargs...))
    end

    if !isempty(optargs)
      @eval begin
        function $(krylov)($(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
          start_time = time_ns()
          workspace = $workspace($(args...))
          warm_start!(workspace, $(optargs...))
          elapsed_time = ktimer(start_time)
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return results(workspace)
        end

        krylov_solve(::Val{Symbol($krylov)}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...), $(optargs...); $(kwargs...))
      end
    end
  else  # --> !isempty(kwargs_workspace)
    @eval begin
      function $(krylov)($(def_args...); $(def_kwargs_workspace...), $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
        start_time = time_ns()
        workspace = $workspace($(args...); $(kwargs_workspace...))
        elapsed_time = ktimer(start_time)
        timemax -= elapsed_time
        $(krylov!)(workspace, $(args...); $(kwargs...))
        workspace.stats.timer += elapsed_time
        return results(workspace)
      end

      krylov_solve(::Val{Symbol($krylov)}, $(def_args...); $(def_kwargs_workspace...), $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...); $(kwargs_workspace...), $(kwargs...))
    end

    if !isempty(optargs)
      @eval begin
        function $(krylov)($(def_args...), $(def_optargs...); $(def_kwargs_workspace...), $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
          start_time = time_ns()
          workspace = $workspace($(args...); $(kwargs_workspace...))
          warm_start!(workspace, $(optargs...))
          elapsed_time = ktimer(start_time)
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return results(workspace)
        end

        krylov_solve(::Val{Symbol($krylov)}, $(def_args...), $(def_optargs...); $(def_kwargs_workspace...), $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} = $(krylov)($(args...), $(optargs...); $(kwargs_workspace...), $(kwargs...))
      end
    end
  end

  ## In-place
  @eval krylov_solve!(workspace :: $workspace{T,FC,SV,SM}, $(def_args...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}} = $(krylov!)(workspace, $(args...); $(kwargs...))

  if !isempty(optargs)
    for krylov_ip in (:krylov_solve!, krylov!)
      @eval begin
        function $(krylov_ip)(workspace :: $workspace{T,FC,SV,SM}, $(def_args...), $(def_optargs...); $(def_kwargs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}}
          start_time = time_ns()
          warm_start!(workspace, $(optargs...))
          elapsed_time = start_time |> ktimer
          timemax -= elapsed_time
          $(krylov!)(workspace, $(args...); $(kwargs...))
          workspace.stats.timer += elapsed_time
          return workspace
        end
      end
    end
  end
end

# Block Krylov methods
for (KS, fun, nsol, nA, nAt, warm_start) in [
  (:BlockCgWorkspace    , :block_cg!    , 1, 1, 0, true)
]
  @eval begin
    elapsed_time(workspace :: $KS) = workspace.stats.timer
    elapsed_allocation_time(workspace :: $KS) = workspace.stats.allocation_timer
    statistics(workspace :: $KS) = workspace.stats
    solution_count(workspace :: $KS) = $nsol
    iteration_count(workspace :: $KS) = workspace.stats.niter
    Aprod_count(workspace :: $KS) = $nA * workspace.stats.niter
    Atprod_count(workspace :: $KS) = $nAt * workspace.stats.niter
    if $nsol == 1
      solution(workspace :: $KS) = workspace.X
      solution(workspace :: $KS, p :: Integer) = (p == 1) ? solution(workspace) : error("solution(workspace) has only one output.")
      results(workspace :: $KS) = (workspace.X, workspace.stats)
    end
    issolved(workspace :: $KS) = workspace.stats.solved
    if $warm_start
      function warm_start!(workspace :: $KS, X0)
        n2, p2 = size(X0)
        SM = typeof(workspace.X)
        (workspace.n == n2 && workspace.p == p2) || error("X0 should have size ($(workspace.n), $(workspace.p))")
        allocate_if(true, workspace, :ΔX, SM, workspace.n, workspace.p)
        copyto!(workspace.ΔX, X0)
        workspace.warm_start = true
        return workspace
      end
    end
  end
end

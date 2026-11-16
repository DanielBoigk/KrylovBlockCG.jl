# Krylov.jl's generic interface and workspace accessors with `block_cg`, as Krylov.jl tests
# them for its own block methods (`test/test_interface.jl`).
@testset "generic interface" begin
  for FC in (Float32, Float64, ComplexF32, ComplexF64), use_val in (true, false)
    @testset "Data Type: $FC, Val: $use_val" begin
      A = FC.(get_div_grad(4, 4, 4))
      m, n = size(A)
      p = 8
      B = rand(FC, m, p)
      SV = Vector{FC}
      SM = Matrix{FC}
      workspace = use_val ? @inferred(krylov_workspace(Val(:block_cg), m, n, p, SV, SM)) :
                            krylov_workspace(:block_cg, m, n, p, SV, SM)
      @test workspace isa BlockCgWorkspace
      @test workspace isa BlockKrylovWorkspace
      for i = 1 : 3
        B = 5 * B
        use_val ? @inferred(krylov_solve(Val(:block_cg), A, B)) : krylov_solve(:block_cg, A, B)
        @inferred krylov_solve!(workspace, A, B)
        niter = iteration_count(workspace)
        @test Aprod_count(workspace) == niter
        @test Atprod_count(workspace) == 0
        @test solution(workspace) === workspace.X
        @test solution_count(workspace) == 1
        @test niter > 0
        @test statistics(workspace) === workspace.stats
        @test issolved(workspace)
        @test elapsed_allocation_time(workspace) > 0
        @test elapsed_time(workspace) > 0
      end
      io = IOBuffer()
      show(io, workspace, show_stats=false)
      show(io, workspace, show_stats=true)
      @test !isempty(String(take!(io)))
    end
  end
end

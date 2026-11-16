# Allocations of `block_cg`, as in the allocation tests of Krylov.jl (`test/test_allocations.jl`).
@testset "allocations" begin
  for FC in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "Data Type: $FC" begin
      A   = FC.(get_div_grad(18, 18, 18))  # Dimension m x n
      m,n = size(A)
      p   = 5
      B   = A * Matrix{FC}(I, m, p)  # Dimension m × p
      T = real(FC)
      nbits_FC = sizeof(FC)
      nbits_T = sizeof(T)

      @testset "BLOCK-CG" begin
        # BLOCK-CG needs (without a null space):
        # - 5 (n*p)-matrices: X, R, Z, P, Q
        # - 2 (p*p)-matrices: G, C (device) and 6 on the host: Gh, Ch, Th, Eh, Lh, Wh
        # - 2 (1*p)-matrices: mask, colsums (device) and maskh, colsumsh (host)
        # - 5 p-vectors of T: d, λh, bnorm, rnorm, tol; 1 p-vector of Int: perm; 1 of Bool: active
        function storage_block_cg_bytes(n, p)
          res = 5*n*p + 8*p*p + 4*p
          return nbits_FC * res + nbits_T * 5 * p + sizeof(Int) * p + p
        end

        workspace = BlockCgWorkspace(A, B)
        block_cg!(workspace, A, B)  # warmup
        inplace_block_cg_bytes = @allocated block_cg!(workspace, A, B)
        @test inplace_block_cg_bytes == 0

        # with a preconditioner and a null space (here: a trivial one, the projection is exact)
        V = zeros(FC, n); V[1] = 1
        workspace = BlockCgWorkspace(A, B; nullspace = V)
        Md = Diagonal(ones(FC, n))
        preconditioned_block_cg!(workspace, A, B, M) = block_cg!(workspace, A, B; M)  # (keyword calls at top level box their arguments)
        preconditioned_block_cg!(workspace, A, B, Md)  # warmup
        inplace_block_cg_bytes = @allocated preconditioned_block_cg!(workspace, A, B, Md)
        @test inplace_block_cg_bytes == 0

        expected_block_cg_bytes = storage_block_cg_bytes(n, p)
        block_cg(A, B)  # warmup
        actual_block_cg_bytes = @allocated block_cg(A, B)
        if VERSION < v"1.11.5" || !Sys.isapple()
          @test expected_block_cg_bytes ≤ actual_block_cg_bytes ≤ 1.05 * expected_block_cg_bytes
        end
      end
    end
  end
end

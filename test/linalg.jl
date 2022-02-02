using Test: @test, @testset
import LinearAlgebra
import RestrictedBoltzmannMachines as RBMs

@testset "linalg" begin
    A = randn(2,2) + LinearAlgebra.Diagonal([10, 10])
    B = randn(2,2)
    C = randn(2,2)
    D = randn(2,2) + LinearAlgebra.Diagonal([10, 10])
    @test RBMs.block_matrix_logdet(A, B, C, D) ≈ LinearAlgebra.logdet([A B; C D])
    @test RBMs.block_matrix_invert(A, B, C, D) ≈ inv([A B; C D])

    A = randn(3,3) + LinearAlgebra.Diagonal([10, 10, 10])
    B = randn(3,2)
    C = randn(2,3)
    D = randn(2,2) + LinearAlgebra.Diagonal([10, 10])
    @test RBMs.block_matrix_logdet(A, B, C, D) ≈ LinearAlgebra.logdet([A B; C D])
    @test RBMs.block_matrix_invert(A, B, C, D) ≈ inv([A B; C D])

    A = randn(2,2) + LinearAlgebra.Diagonal([10, 10])
    B = randn(2,3)
    C = randn(3,2)
    D = randn(3,3) + LinearAlgebra.Diagonal([10, 10, 10])
    @test RBMs.block_matrix_logdet(A, B, C, D) ≈ LinearAlgebra.logdet([A B; C D])
    @test RBMs.block_matrix_invert(A, B, C, D) ≈ inv([A B; C D])
end

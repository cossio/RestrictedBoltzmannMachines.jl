include("tests_init.jl")

@testset "linalg" begin
    A = randn(2,2)
    B = randn(2,2)
    C = randn(2,2)
    D = randn(2,2)
    M = [A B; C D]
    RBMs.block_matrix_logdet(A, B, C, D) ≈ logdet([A B; C D])
end

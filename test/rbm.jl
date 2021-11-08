include("tests_init.jl")

@testset "RBM" begin
    rbm = RBM(Binary(5, 2), Binary(4, 3), randn(5, 2, 4, 3))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    v = rand(Bool, size(rbm.visible)..., 7)
    h = rand(Bool, size(rbm.hidden)..., 7)
    @test size(@inferred interaction_energy(rbm, v, h)) == (7,)
    @test size(@inferred energy(rbm, v, h)) == (7,)
    Ew = -[sum(v[i,j,b] * rbm.weights[i,j,μ,ν] * h[μ,ν,b] for i=1:5, j=1:2, μ=1:4, ν=1:3) for b=1:7]
    @test interaction_energy(rbm, v, h) ≈ Ew
    @test energy(rbm, v, h) ≈ energy(rbm.visible, v) + energy(rbm.hidden, h) + Ew
    @test size(@inferred sample_h_from_v(rbm, v)) == size(h)
    @test size(@inferred sample_v_from_h(rbm, h)) == size(v)
    @test size(@inferred sample_v_from_v(rbm, v)) == size(v)
    @test size(@inferred sample_h_from_h(rbm, h)) == size(h)
    @test size(@inferred free_energy(rbm, v)) == (7,)
    @test size(@inferred reconstruction_error(rbm, v)) == (7,)
end

@testset "Gaussian-Gaussian RBM" begin
    N, M = 2, 3
    rbm = RBM(Gaussian(N), Gaussian(M), reshape(1:(N*M), N, M) / (10*N*M))
    rbm.visible.γ .= randperm(N)
    rbm.hidden.γ .= randperm(M)
    rbm.visible.θ .= randperm(N)
    rbm.hidden.θ .= randperm(M)
    J = [diagm(rbm.visible.γ) rbm.weights;
         rbm.weights'  diagm(rbm.hidden.γ)]
    @test log_partition(rbm) ≈ (N + M)/2 * log(2π) - logdet(J)/2

    v = reshape(1:N, :, 1)
    Ev = sum(@. rbm.visible.γ * v^2 / 2 - rbm.visible.θ * v)
    Γv = sum((rbm.hidden.θ .+ rbm.weights' * v).^2 ./ 2rbm.hidden.γ)
    @test free_energy(rbm, v)[1] ≈ Ev - Γv - sum(log.(2π ./ rbm.hidden.γ)) / 2

    v = reshape((1:N) .- M, :, 1)
    @test log_likelihood(rbm, v) ≈ -free_energy(rbm, v) .- log_partition(rbm)
end

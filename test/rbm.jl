include("tests_init.jl")

@testset "RBM" begin
    rbm = RBMs.RBM(RBMs.Binary(5, 2), RBMs.Binary(4, 3), randn(5, 2, 4, 3))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)

    v = rand(Bool, size(rbm.visible)..., 7)
    h = rand(Bool, size(rbm.hidden)..., 7)
    @test size(@inferred RBMs.interaction_energy(rbm, v, h)) == (7,)
    @test size(@inferred RBMs.energy(rbm, v, h)) == (7,)

    Ew = -[sum(v[i,j,b] * rbm.weights[i,j,μ,ν] * h[μ,ν,b] for i=1:5, j=1:2, μ=1:4, ν=1:3) for b=1:7]
    @test RBMs.interaction_energy(rbm, v, h) ≈ Ew
    @test RBMs.energy(rbm, v, h) ≈ RBMs.energy(rbm.visible, v) + RBMs.energy(rbm.hidden, h) + Ew

    @test size(@inferred RBMs.sample_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.sample_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.sample_v_from_v(rbm, v)) == size(v)
    @test size(@inferred RBMs.sample_h_from_h(rbm, h)) == size(h)

    @test size(@inferred RBMs.free_energy(rbm, v)) == (7,)
    @test size(@inferred RBMs.reconstruction_error(rbm, v)) == (7,)
end

@testset "Gaussian-Gaussian RBM" begin
    N, M = 2, 3
    rbm = RBMs.RBM(RBMs.Gaussian(N), RBMs.Gaussian(M), reshape(1:(N*M), N, M) / (10*N*M))
    rbm.visible.γ .= randperm(N)
    rbm.hidden.γ .= randperm(M)
    rbm.visible.θ .= randperm(N)
    rbm.hidden.θ .= randperm(M)

    θ = [rbm.visible.θ; rbm.hidden.θ]

    A = [Diagonal(rbm.visible.γ) rbm.weights;
         rbm.weights'  Diagonal(rbm.hidden.γ)]
    @test RBMs.log_partition(rbm) ≈ (N + M)/2 * log(2π) + θ' * inv(A) * θ / 2 - logdet(A)/2

    v = reshape(1:N, :, 1)
    Ev = sum(@. rbm.visible.γ * v^2 / 2 - rbm.visible.θ * v)
    Γv = sum((rbm.hidden.θ .+ rbm.weights' * v).^2 ./ 2rbm.hidden.γ)
    @test RBMs.free_energy(rbm, v)[1] ≈ Ev - Γv - sum(log.(2π ./ rbm.hidden.γ)) / 2

    v = reshape((1 - M):(N - M), :, 1)
    @test RBMs.log_likelihood(rbm, v) ≈ -RBMs.free_energy(rbm, v) .- RBMs.log_partition(rbm)
end

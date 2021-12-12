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

@testset "Gaussian-Gaussian RBM, 1-dimension" begin
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(1), rand(1) .+ 0.5),
        RBMs.Gaussian(randn(1), rand(1) .+ 0.5),
        randn(1, 1) / 1e2
    )

    @test RBMs.log_partition(rbm, 1) ≈ RBMs.log_partition(rbm)

    logZ, ϵ = QuadGK.quadgk(x -> exp(-only(RBMs.free_energy(rbm, [x;;]))), -Inf, Inf)
    @test RBMs.log_partition(rbm) ≈ logZ

    β = 1.5
    logZ, ϵ = QuadGK.quadgk(x -> exp(-β * only(RBMs.free_energy(rbm, [x;;], β))), -Inf, Inf)
    @test RBMs.log_partition(rbm, β) ≈ logZ
end

@testset "Gaussian-Gaussian RBM, multi-dimensional" begin
    N = (10, 3)
    M = (7, 2)
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(N...), rand(N...) .+ 0.5),
        RBMs.Gaussian(randn(M...), rand(M...) .+ 0.5),
        randn(N..., M...) / (10 * prod(N) * prod(M)))

    θ = [
        vec(rbm.visible.θ);
        vec(rbm.hidden.θ)
    ]
    γv = vec(abs.(rbm.visible.γ))
    γh = vec(abs.(rbm.hidden.γ))
    w = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    A = [diagm(γv) -w;
         -w'  diagm(γh)]

    v = randn(N..., 1)
    h = randn(M..., 1)
    x = [
        reshape(v, length(rbm.visible), 1);
        reshape(h, length(rbm.hidden),  1)
    ]

    @test RBMs.energy(rbm, v, h) ≈ x' * A * x / 2 - θ' * x
    @test RBMs.log_partition(rbm) ≈ (
        (prod(N) + prod(M))/2 * log(2π) + θ' * inv(A) * θ / 2 - logdet(A)/2
    )
    @test RBMs.log_partition(rbm, 1) ≈ RBMs.log_partition(rbm)

    β = rand()
    @test RBMs.log_partition(rbm, β) ≈ (
        (prod(N) + prod(M))/2 * log(2π) + β^2 * θ' * inv(β*A) * θ / 2 - logdet(β*A)/2
    )

    @test RBMs.log_likelihood(rbm, v, 1) ≈ @test RBMs.log_likelihood(rbm, v)
    @test RBMs.log_likelihood(rbm, v, β) ≈ (
        -RBMs.free_energy(rbm, v, β) .- RBMs.log_partition(rbm, β)
    )

    Ev = sum(@. rbm.visible.γ * v^2 / 2 - rbm.visible.θ * v)
    Γv = sum((rbm.hidden.θ .+ rbm.weights' * v).^2 ./ 2rbm.hidden.γ)
    @test RBMs.free_energy(rbm, v)[1] ≈ Ev - Γv - sum(log.(2π ./ rbm.hidden.γ)) / 2

    c = inv(A) # covariances
    μ = c * θ  # means

    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
        RBMs.log_partition(rbm)
    end
    ∂θv = vec(gs[rbm.visible.θ])
    ∂θh = vec(gs[rbm.hidden.θ])
    ∂γv = vec(abs.(gs[rbm.visible.γ]))
    ∂γh = vec(abs.(gs[rbm.hidden.γ]))
    ∂w = reshape(gs[rbm.weights], length(rbm.visible), length(rbm.hidden))
    @test μ ≈ [∂θv; ∂θh]
    @test c + μ * μ' ≈ [
        diagm(∂γv)  -∂w
        -∂w'  diagm(∂γh)
    ]
end

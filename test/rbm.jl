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

    @inferred RBMs.flip_layers(rbm)

    @inferred RBMs.conjugates_h_from_v(rbm, v)
    @inferred RBMs.conjugates_v_from_h(rbm, h)
end

@testset "Gaussian-Gaussian RBM, 1-dimension" begin
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(1), rand(1) .+ 1),
        RBMs.Gaussian(randn(1), rand(1) .+ 1),
        randn(1, 1) / 1e2
    )

    @assert isposdef([
        rbm.visible.γ -rbm.weights;
        -rbm.weights'  rbm.hidden.γ
    ])

    @test RBMs.log_partition(rbm, 1) ≈ RBMs.log_partition(rbm)

    Z, ϵ = QuadGK.quadgk(x -> exp(-only(RBMs.free_energy(rbm, [x;;]))), -Inf, Inf)
    @test RBMs.log_partition(rbm) ≈ log(Z)

    β = 1.5
    Z, ϵ = QuadGK.quadgk(x -> exp(-β * only(RBMs.free_energy(rbm, [x;;], β))), -Inf, Inf)
    @test RBMs.log_partition(rbm, β) ≈ log(Z)
end

@testset "Gaussian-Gaussian RBM, multi-dimensional" begin
    n = (10, 3)
    m = (7, 2)
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(n...), rand(n...) .+ 0.5),
        RBMs.Gaussian(randn(m...), rand(m...) .+ 0.5),
        randn(n..., m...) / (10 * prod(n) * prod(m)))

    N = length(rbm.visible)
    M = length(rbm.hidden)

    θ = [
        vec(rbm.visible.θ);
        vec(rbm.hidden.θ)
    ]
    γv = vec(rbm.visible.γ)
    γh = vec(rbm.hidden.γ)
    w = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    A = [diagm(γv) -w;
         -w'  diagm(γh)]

    v = randn(n..., 1)
    h = randn(m..., 1)
    x = [reshape(v, N, 1); reshape(h, M, 1)]

    @test RBMs.energy(rbm, v, h) ≈ x' * A * x / 2 - θ' * x
    @test RBMs.log_partition(rbm) ≈ (
        (N + M)/2 * log(2π) + θ' * inv(A) * θ / 2 - logdet(A)/2
    )
    @test RBMs.log_partition(rbm, 1) ≈ RBMs.log_partition(rbm)

    β = rand()
    @test RBMs.log_partition(rbm, β) ≈ (
        (N + M)/2 * log(2π) + β * θ' * inv(A) * θ / 2 - logdet(β*A)/2
    ) rtol=1e-6

    @test RBMs.log_likelihood(rbm, v, 1) ≈ RBMs.log_likelihood(rbm, v)
    @test RBMs.log_likelihood(rbm, v, β) ≈ (
        -β * RBMs.free_energy(rbm, v, β) .- RBMs.log_partition(rbm, β)
    )

    Ev = sum(@. rbm.visible.γ * v^2 / 2 - rbm.visible.θ * v)
    Γv = sum((rbm.hidden.θ .+ RBMs.inputs_v_to_h(rbm, v)).^2 ./ 2rbm.hidden.γ)
    @test only(RBMs.free_energy(rbm, v)) ≈ Ev - Γv - sum(log.(2π ./ rbm.hidden.γ)) / 2

    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
        RBMs.log_partition(rbm)
    end
    ∂θv = vec(gs[rbm.visible.θ])
    ∂θh = vec(gs[rbm.hidden.θ])
    ∂γv = vec(gs[rbm.visible.γ])
    ∂γh = vec(gs[rbm.hidden.γ])
    ∂w = reshape(gs[rbm.weights], length(rbm.visible), length(rbm.hidden))

    Σ = inv(A) # covariances
    μ = Σ * θ  # means
    C = Σ + μ * μ' # non-centered second moments

    @test [∂θv; ∂θh] ≈ μ
    @test -2∂γv ≈ diag(C[1:N, 1:N]) # <vi^2>
    @test -2∂γh ≈ diag(C[(N + 1):end, (N + 1):end]) # <hμ^2>
    @test ∂w ≈ C[1:N, (N + 1):end] # <vi*hμ>
end

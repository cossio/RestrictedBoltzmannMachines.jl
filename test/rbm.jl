include("tests_init.jl")

@testset "flat_interaction_energy" begin
    N, M, B = 21, 13, 5

    w = randn(N, M)
    v = randn(N)
    h = randn(M)
    @test RBMs.flat_interaction_energy(w, v, h) ≈ -v' * w * h
    @test RBMs.flat_interaction_energy(w, v, h) ≈ RBMs.flat_interaction_energy(w', h, v)

    w = randn(N, M)
    v = randn(N, 1)
    h = randn(M, 1)
    @test RBMs.flat_interaction_energy(w, vec(v), vec(h)) ≈ -vec(v)' * w * vec(h)
    @test RBMs.flat_interaction_energy(w, v, h) ≈ -[vec(v)' * w * vec(h)]
    @test RBMs.flat_interaction_energy(w, v, h) ≈ RBMs.flat_interaction_energy(w', h, v)

    w = randn(N, M)
    v = randn(N, B)
    h = randn(M, B)
    @test RBMs.flat_interaction_energy(w, v, h) ≈ -diag(v' * w * h)
    @test RBMs.flat_interaction_energy(w, v, h) ≈ RBMs.flat_interaction_energy(w', h, v)
end

@testset "Binary-Binary RBM" begin
    rbm = RBMs.RBM(RBMs.Binary(5, 2), RBMs.Binary(4, 3), randn(5, 2, 4, 3))
    randn!(rbm.w)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)

    v = rand(Bool, size(rbm.visible)..., 7)
    h = rand(Bool, size(rbm.hidden)..., 7)
    @test size(@inferred RBMs.interaction_energy(rbm, v, h)) == (7,)
    @test size(@inferred RBMs.energy(rbm, v, h)) == (7,)

    Ew = -[sum(v[i,j,b] * rbm.w[i,j,μ,ν] * h[μ,ν,b] for i=1:5, j=1:2, μ=1:4, ν=1:3) for b=1:7]
    @test RBMs.interaction_energy(rbm, v, h) ≈ Ew
    @test RBMs.energy(rbm, v, h) ≈ RBMs.energy(rbm.visible, v) + RBMs.energy(rbm.hidden, h) + Ew

    @test (@inferred RBMs.energy(rbm, v[:,:,1], h[:,:,1])) isa Real
    @test (@inferred RBMs.energy(rbm, v[:,:,1], h)) isa AbstractVector{<:Real}
    @test (@inferred RBMs.energy(rbm, v, h[:,:,1])) isa AbstractVector{<:Real}

    for b = 1:7
        @test RBMs.energy(rbm, v[:,:,b], h[:,:,b]) ≈ RBMs.energy(rbm, v, h)[b]
        @test RBMs.energy(rbm, v[:,:,b], h) ≈ [RBMs.energy(rbm, v[:,:,b], h[:,:,k]) for k=1:7]
        @test RBMs.energy(rbm, v, h[:,:,b]) ≈ [RBMs.energy(rbm, v[:,:,k], h[:,:,b]) for k=1:7]
    end

    @test size(@inferred RBMs.sample_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.sample_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.sample_h_from_v(rbm, v[:,:,1])) == size(rbm.hidden)
    @test size(@inferred RBMs.sample_v_from_h(rbm, h[:,:,1])) == size(rbm.visible)
    for k = 1:3
        @test size(@inferred RBMs.sample_v_from_v(rbm, v; steps=k)) == size(v)
        @test size(@inferred RBMs.sample_h_from_h(rbm, h; steps=k)) == size(h)
        @test size(@inferred RBMs.sample_v_from_v(rbm, v[:,:,1]; steps=k)) == size(rbm.visible)
        @test size(@inferred RBMs.sample_h_from_h(rbm, h[:,:,1]; steps=k)) == size(rbm.hidden)
    end

    @test size(@inferred RBMs.mean_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.mean_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.mean_h_from_v(rbm, v[:,:,1])) == size(rbm.hidden)
    @test size(@inferred RBMs.mean_v_from_h(rbm, h[:,:,1])) == size(rbm.visible)

    @test size(@inferred RBMs.mode_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.mode_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.mode_h_from_v(rbm, v[:,:,1])) == size(rbm.hidden)
    @test size(@inferred RBMs.mode_v_from_h(rbm, h[:,:,1])) == size(rbm.visible)

    @test size(@inferred RBMs.free_energy(rbm, v)) == (7,)
    @test size(@inferred RBMs.reconstruction_error(rbm, v)) == (7,)
    @test (@inferred RBMs.free_energy(rbm, v[:,:,1])) isa Real
    @test (@inferred RBMs.reconstruction_error(rbm, v[:,:,1])) isa Real

    @inferred RBMs.mirror(rbm)
    @test RBMs.mirror(rbm).visible == rbm.hidden
    @test RBMs.mirror(rbm).hidden == rbm.visible
    @test RBMs.energy(RBMs.mirror(rbm), h, v) ≈ RBMs.energy(rbm, v, h)

    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
        mean(RBMs.free_energy(rbm, v))
    end
    ∂F = RBMs.∂free_energy(rbm, v)
    @test ∂F.visible.θ ≈ gs[rbm.visible.θ]
    @test ∂F.hidden.θ ≈ gs[rbm.hidden.θ]
    @test ∂F.w ≈ gs[rbm.w]

    v1 = rand(Bool, size(rbm.visible)..., 7)
    v2 = rand(Bool, size(rbm.visible)..., 7)
    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
        RBMs.contrastive_divergence(rbm, v1, v2)
    end
    ∂F = RBMs.∂contrastive_divergence(rbm, v1, v2)
    @test ∂F.visible.θ ≈ gs[rbm.visible.θ]
    @test ∂F.hidden.θ ≈ gs[rbm.hidden.θ]
    @test ∂F.w ≈ gs[rbm.w]
end

@testset "Gaussian-Gaussian RBM, 1-dimension" begin
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(1), rand(1) .+ 1),
        RBMs.Gaussian(randn(1), rand(1) .+ 1),
        randn(1, 1) / 1e2
    )

    @assert isposdef([
        rbm.visible.γ -rbm.w;
        -rbm.w'  rbm.hidden.γ
    ])

    @test RBMs.log_partition(rbm; β = 1) ≈ RBMs.log_partition(rbm)

    Z, ϵ = QuadGK.quadgk(x -> exp(-only(RBMs.free_energy(rbm, [x;;]))), -Inf, Inf)
    @test RBMs.log_partition(rbm) ≈ log(Z)

    β = 1.5
    Z, ϵ = QuadGK.quadgk(x -> exp(-β * only(RBMs.free_energy(rbm, [x;;]; β))), -Inf, Inf)
    @test RBMs.log_partition(rbm; β) ≈ log(Z)
end

@testset "Gaussian-Gaussian RBM, multi-dimensional" begin
    n = (10, 3)
    m = (7, 2)
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(n...), rand(n...) .+ 0.5),
        RBMs.Gaussian(randn(m...), rand(m...) .+ 0.5),
        randn(n..., m...) / (10 * prod(n) * prod(m))
    )

    N = length(rbm.visible)
    M = length(rbm.hidden)

    θ = [
        vec(rbm.visible.θ);
        vec(rbm.hidden.θ)
    ]
    γv = vec(rbm.visible.γ)
    γh = vec(rbm.hidden.γ)
    w = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    A = [diagm(γv) -w;
         -w'  diagm(γh)]

    v = randn(n..., 1)
    h = randn(m..., 1)
    x = [reshape(v, N, 1); reshape(h, M, 1)]

    @test RBMs.energy(rbm, v, h) ≈ x' * A * x / 2 - θ' * x
    @test RBMs.log_partition(rbm) ≈ (
        (N + M)/2 * log(2π) + θ' * inv(A) * θ / 2 - logdet(A)/2
    )
    @test RBMs.log_partition(rbm; β = 1) ≈ RBMs.log_partition(rbm)

    β = rand()
    @test RBMs.log_partition(rbm; β) ≈ (
        (N + M)/2 * log(2π) + β * θ' * inv(A) * θ / 2 - logdet(β*A)/2
    ) rtol=1e-6

    @test RBMs.log_likelihood(rbm, v; β = 1) ≈ RBMs.log_likelihood(rbm, v)
    @test RBMs.log_likelihood(rbm, v; β) ≈ (
        -β * RBMs.free_energy(rbm, v; β) .- RBMs.log_partition(rbm; β)
    )

    Ev = sum(@. rbm.visible.γ * v^2 / 2 - rbm.visible.θ * v)
    Fv = sum(-(rbm.hidden.θ .+ RBMs.inputs_v_to_h(rbm, v)).^2 ./ 2rbm.hidden.γ)
    @test only(RBMs.free_energy(rbm, v)) ≈ Ev + Fv - sum(log.(2π ./ rbm.hidden.γ)) / 2

    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
        RBMs.log_partition(rbm)
    end
    ∂θv = vec(gs[rbm.visible.θ])
    ∂θh = vec(gs[rbm.hidden.θ])
    ∂γv = vec(gs[rbm.visible.γ])
    ∂γh = vec(gs[rbm.hidden.γ])
    ∂w = reshape(gs[rbm.w], length(rbm.visible), length(rbm.hidden))

    Σ = inv(A) # covariances
    μ = Σ * θ  # means
    C = Σ + μ * μ' # non-centered second moments

    @test [∂θv; ∂θh] ≈ μ
    @test -2∂γv ≈ diag(C[1:N, 1:N]) # <vi^2>
    @test -2∂γh ≈ diag(C[(N + 1):end, (N + 1):end]) # <hμ^2>
    @test ∂w ≈ C[1:N, (N + 1):end] # <vi*hμ>
end

using EllipsisNotation: (..)
using LinearAlgebra: diag
using LinearAlgebra: diagm
using LinearAlgebra: dot
using LinearAlgebra: isposdef
using LinearAlgebra: logdet
using LogExpFunctions: logsumexp
using QuadGK: quadgk
using Random: bitrand
using Random: randn!
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: ∂free_energy_h
using RestrictedBoltzmannMachines: ∂free_energy_v
using RestrictedBoltzmannMachines: ∂interaction_energy
using RestrictedBoltzmannMachines: batch_size
using RestrictedBoltzmannMachines: batchmean
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: cgf
using RestrictedBoltzmannMachines: energy
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: free_energy_h
using RestrictedBoltzmannMachines: free_energy_v
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: GaussianRBM
using RestrictedBoltzmannMachines: hidden_cgf
using RestrictedBoltzmannMachines: HopfieldRBM
using RestrictedBoltzmannMachines: inputs_h_from_v
using RestrictedBoltzmannMachines: join_batch_size
using RestrictedBoltzmannMachines: inputs_v_from_h
using RestrictedBoltzmannMachines: interaction_energy
using RestrictedBoltzmannMachines: log_likelihood
using RestrictedBoltzmannMachines: log_partition
using RestrictedBoltzmannMachines: mean_from_inputs
using RestrictedBoltzmannMachines: mean_h_from_v
using RestrictedBoltzmannMachines: mean_v_from_h
using RestrictedBoltzmannMachines: mirror
using RestrictedBoltzmannMachines: mode_h_from_v
using RestrictedBoltzmannMachines: mode_v_from_h
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: PottsGumbel
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: reconstruction_error
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: sample_h_from_h
using RestrictedBoltzmannMachines: sample_h_from_v
using RestrictedBoltzmannMachines: sample_v_from_h
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: Spin
using RestrictedBoltzmannMachines: SpinRBM
using RestrictedBoltzmannMachines: var_from_inputs
using RestrictedBoltzmannMachines: var_h_from_v
using RestrictedBoltzmannMachines: var_v_from_h
using RestrictedBoltzmannMachines: visible_cgf
using RestrictedBoltzmannMachines: wmean
using RestrictedBoltzmannMachines: total_mean_h_from_v
using RestrictedBoltzmannMachines: total_mean_v_from_h
using RestrictedBoltzmannMachines: total_var_h_from_v
using RestrictedBoltzmannMachines: total_var_v_from_h
using RestrictedBoltzmannMachines: total_meanvar_h_from_v
using RestrictedBoltzmannMachines: total_meanvar_v_from_h
using Statistics: mean
using Statistics: var
using Test: @inferred
using Test: @test
using Test: @test_throws
using Test: @testset
using Zygote: gradient

@testset "batches, n=$n, m=$m, Bv=$Bv, Bh=$Bh" for n in (5, (5, 2)), m in (2, (3, 4)), Bv in ((), (3, 2)), Bh in ((), (3, 2))
    rbm = BinaryRBM(randn(n...), randn(m...), randn(n..., m...))
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    v = bitrand(n..., Bv...)
    h = bitrand(m..., Bh...)

    @test batch_size(rbm.visible, v) == Bv
    @test batch_size(rbm.hidden, h) == Bh

    @test size(inputs_h_from_v(rbm, v)) == (size(rbm.hidden)..., Bv...)
    @test size(inputs_v_from_h(rbm, h)) == (size(rbm.visible)..., Bh...)

    if length(Bv) == length(Bh) == 0
        @test interaction_energy(rbm, v, h) isa Number
        @test interaction_energy(rbm, v, h) ≈ -vec(v)' * wmat * vec(h)
        @test energy(rbm, v, h) isa Number
    elseif length(Bv) == 0
        hmat = reshape(h, length(rbm.hidden), :)
        E = -vec(v)' * wmat * hmat
        @test interaction_energy(rbm, v, h) isa AbstractArray
        @test size(interaction_energy(rbm, v, h)) == Bh
        @test interaction_energy(rbm, v, h) ≈ reshape(E, Bh)
        @test size(energy(rbm, v, h)) == Bh
    elseif length(Bh) == 0
        vmat = reshape(v, length(rbm.visible), :)
        E = -vec(h)' * wmat' * vmat
        @test interaction_energy(rbm, v, h) isa AbstractArray
        @test size(interaction_energy(rbm, v, h)) == Bv
        @test interaction_energy(rbm, v, h) ≈ reshape(E, Bv)
        @test size(energy(rbm, v, h)) == Bv
    else
        vmat = reshape(v, length(rbm.visible), :)
        hmat = reshape(h, length(rbm.hidden), :)
        E = -dot.(eachcol(vmat), Ref(wmat), eachcol(hmat))
        @test interaction_energy(rbm, v, h) isa AbstractArray
        @test size(interaction_energy(rbm, v, h)) == Bv == Bh
        @test interaction_energy(rbm, v, h) ≈ reshape(E, Bv)
        @test size(energy(rbm, v, h)) == Bv == Bh
    end
    @inferred inputs_h_from_v(rbm, v)
    @inferred inputs_v_from_h(rbm, h)
    @inferred interaction_energy(rbm, v, h)
    @inferred energy(rbm, v, h)
    gs = gradient(rbm) do rbm
        mean(energy(rbm, v, h))
    end
    ∂w = @inferred ∂interaction_energy(rbm, v, h)
    @test ∂w ≈ only(gs).w
end

@testset "singleton batch dims" begin # behavior useful for ConvolutionalRBMs.jl
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2))

    v = bitrand(3, 1, 2)
    h = bitrand(2, 3, 1)
    @test size(energy(rbm, v, h)) == @inferred(batch_size(rbm, v, h)) == (3, 2)
    @test energy(rbm, v, h) ≈ [energy(rbm, v[:, 1, j], h[:, i, 1]) for i in 1:3, j in 1:2]

    v = bitrand(3, 1, 2)
    h = bitrand(2, 3)
    @test size(energy(rbm, v, h)) == @inferred(batch_size(rbm, v, h)) == (3, 2)
    @test energy(rbm, v, h) ≈ [energy(rbm, v[:, 1, j], h[:, i]) for i in 1:3, j in 1:2]

    v = bitrand(3, 1)
    h = bitrand(2, 3, 2)
    @test size(energy(rbm, v, h)) == @inferred(batch_size(rbm, v, h)) == (3, 2)
    @test energy(rbm, v, h) ≈ [energy(rbm, v[:, 1], h[:, i, j]) for i in 1:3, j in 1:2]

    v = bitrand(3, 1, 2)
    h = bitrand(2, 3, 2)
    @test size(energy(rbm, v, h)) == @inferred(batch_size(rbm, v, h)) == (3, 2)
    @test energy(rbm, v, h) ≈ [energy(rbm, v[:, 1, j], h[:, i, j]) for i in 1:3, j in 1:2]

    v = bitrand(3, 3, 1)
    h = bitrand(2, 3, 2)
    @test size(energy(rbm, v, h)) == @inferred(batch_size(rbm, v, h)) == (3, 2)
    @test energy(rbm, v, h) ≈ [energy(rbm, v[:, i, 1], h[:, i, j]) for i in 1:3, j in 1:2]

    v = bitrand(3, 2, 2)
    h = bitrand(2, 3, 1)
    @test_throws Any batch_size(rbm, v, h)
    @test_throws Any energy(rbm, v, h)

    v = bitrand(3, 1, 2)
    h = bitrand(2, 3, 3)
    @test_throws Any batch_size(rbm, v, h)
    @test_throws Any energy(rbm, v, h)
end

@testset "join_batch_size" begin
    @test join_batch_size((3, 5), (3, 5)) == (3, 5)
    @test join_batch_size((1, 5), (3, 5)) == (3, 5)
    @test join_batch_size((3, 1), (3, 5)) == (3, 5)
    @test join_batch_size((3,), (3, 7)) == (3, 7)
    @test join_batch_size((3, 7), (3,)) == (3, 7)
    @test_throws AssertionError join_batch_size((2, 5), (3, 5))
end

@testset "sample_v_from_v and sample_h_from_h on binary RBM" begin
    rbm = BinaryRBM(randn(3, 2), randn(2, 3), zeros(3, 2, 2, 3))
    v = bitrand(size(rbm.visible)..., 10^6)
    v = sample_v_from_v(rbm, v)
    @test batchmean(rbm.visible, v) ≈ mean_from_inputs(rbm.visible) rtol = 0.1

    h = bitrand(size(rbm.hidden)..., 10^6)
    h = sample_h_from_h(rbm, h)
    @test batchmean(rbm.hidden, h) ≈ mean_from_inputs(rbm.hidden) rtol = 0.1

    randn!(rbm.w)
    h = sample_h_from_v(rbm, v)
    μ = mean_from_inputs(rbm.hidden, inputs_h_from_v(rbm, v))
    @test batchmean(rbm.hidden, h) ≈ batchmean(rbm.hidden, μ) rtol = 0.1

    randn!(rbm.w)
    v = sample_v_from_h(rbm, h)
    μ = mean_from_inputs(rbm.visible, inputs_v_from_h(rbm, h))
    @test batchmean(rbm.visible, v) ≈ batchmean(rbm.visible, μ) rtol = 0.1
end

@testset "rbm convenience constructors" begin
    rbm = BinaryRBM(randn(5), randn(3), randn(5, 3))
    @test rbm.visible isa Binary
    @test rbm.hidden isa Binary
    @test size(rbm.w) == (5, 3)

    rbm = HopfieldRBM(randn(5), randn(3), rand(3), randn(5, 3))
    @test rbm.visible isa Spin
    @test rbm.hidden isa Gaussian
    @test size(rbm.w) == (5, 3)
    @test all(rbm.hidden.γ .> 0)

    rbm = HopfieldRBM(randn(5), randn(5, 3))
    @test rbm.visible isa Spin
    @test iszero(mean_from_inputs(rbm.hidden))
    @test var_from_inputs(rbm.hidden) == ones(size(rbm.hidden))
    @test size(rbm.w) == (5, 3)
end

@testset "Binary-Binary RBM" begin
    rbm = BinaryRBM(randn(5, 2), randn(4, 3), randn(5, 2, 4, 3))
    v = rand(Bool, size(rbm.visible)..., 7)
    h = rand(Bool, size(rbm.hidden)..., 7)
    @test size(@inferred interaction_energy(rbm, v, h)) == (7,)
    @test size(@inferred energy(rbm, v, h)) == (7,)

    Ew = -[sum(v[i, j, b] * rbm.w[i, j, μ, ν] * h[μ, ν, b] for i in 1:5, j in 1:2, μ in 1:4, ν in 1:3) for b in 1:7]
    @test interaction_energy(rbm, v, h) ≈ Ew
    @test energy(rbm, v, h) ≈ energy(rbm.visible, v) + energy(rbm.hidden, h) + Ew

    @test (@inferred energy(rbm, v[:, :, 1], h[:, :, 1])) isa Real
    @test (@inferred energy(rbm, v[:, :, 1], h)) isa AbstractVector{<:Real}
    @test (@inferred energy(rbm, v, h[:, :, 1])) isa AbstractVector{<:Real}

    for b in 1:7
        @test energy(rbm, v[:, :, b], h[:, :, b]) ≈ energy(rbm, v, h)[b]
        @test energy(rbm, v[:, :, b], h) ≈ [energy(rbm, v[:, :, b], h[:, :, k]) for k in 1:7]
        @test energy(rbm, v, h[:, :, b]) ≈ [energy(rbm, v[:, :, k], h[:, :, b]) for k in 1:7]
    end

    @test size(@inferred sample_h_from_v(rbm, v)) == size(h)
    @test size(@inferred sample_v_from_h(rbm, h)) == size(v)
    @test size(@inferred sample_h_from_v(rbm, v[:, :, 1])) == size(rbm.hidden)
    @test size(@inferred sample_v_from_h(rbm, h[:, :, 1])) == size(rbm.visible)
    for k in 1:3
        @test size(@inferred sample_v_from_v(rbm, v; steps = k)) == size(v)
        @test size(@inferred sample_h_from_h(rbm, h; steps = k)) == size(h)
        @test size(@inferred sample_v_from_v(rbm, v[:, :, 1]; steps = k)) == size(rbm.visible)
        @test size(@inferred sample_h_from_h(rbm, h[:, :, 1]; steps = k)) == size(rbm.hidden)
    end

    @test size(@inferred mean_h_from_v(rbm, v)) == size(h)
    @test size(@inferred mean_v_from_h(rbm, h)) == size(v)
    @test size(@inferred mean_h_from_v(rbm, v[:, :, 1])) == size(rbm.hidden)
    @test size(@inferred mean_v_from_h(rbm, h[:, :, 1])) == size(rbm.visible)

    @test size(@inferred mode_h_from_v(rbm, v)) == size(h)
    @test size(@inferred mode_v_from_h(rbm, h)) == size(v)
    @test size(@inferred mode_h_from_v(rbm, v[:, :, 1])) == size(rbm.hidden)
    @test size(@inferred mode_v_from_h(rbm, h[:, :, 1])) == size(rbm.visible)

    @test size(@inferred var_h_from_v(rbm, v)) == size(h)
    @test size(@inferred var_v_from_h(rbm, h)) == size(v)
    @test size(@inferred var_h_from_v(rbm, v[:, :, 1])) == size(rbm.hidden)
    @test size(@inferred var_v_from_h(rbm, h[:, :, 1])) == size(rbm.visible)

    @test size(@inferred free_energy(rbm, v)) == (7,)
    @test size(@inferred reconstruction_error(rbm, v)) == (7,)
    @test (@inferred free_energy(rbm, v[:, :, 1])) isa Real
    @test (@inferred reconstruction_error(rbm, v[:, :, 1])) isa Real
end

@testset "mirror" begin
    rbm = BinaryRBM(randn(5, 2), randn(7, 4, 3), randn(5, 2, 7, 4, 3))
    v = rand(Bool, size(rbm.visible)..., 13)
    h = rand(Bool, size(rbm.hidden)..., 13)
    @inferred mirror(rbm)
    @test mirror(rbm).visible == rbm.hidden
    @test mirror(rbm).hidden == rbm.visible
    @test energy(mirror(rbm), h, v) ≈ energy(rbm, v, h)

    @test free_energy(rbm, v) ≈ @inferred free_energy_v(rbm, v)
    @test free_energy(mirror(rbm), h) ≈ @inferred free_energy_h(rbm, h)
end

@testset "mirror with Potts-family layers" begin
    rbm = RBM(
        Potts(; θ = randn(3, 4, 2)),
        PottsGumbel(; θ = randn(5, 3)),
        randn(3, 4, 2, 5, 3),
    )
    v = sample_from_inputs(rbm.visible, zeros(size(rbm.visible)..., 11))
    h = sample_from_inputs(rbm.hidden, zeros(size(rbm.hidden)..., 11))
    rbm_mirror = @inferred mirror(rbm)
    @test rbm_mirror.visible == rbm.hidden
    @test rbm_mirror.hidden == rbm.visible
    @test rbm_mirror.w ≈ permutedims(rbm.w, (4, 5, 1, 2, 3))
    @test energy(rbm_mirror, h, v) ≈ energy(rbm, v, h)
end

@testset "binary free energy" begin
    rbm = BinaryRBM(randn(1), randn(1), randn(1, 1))
    for v in [[0], [1]]
        @test -free_energy(rbm, v) ≈ logsumexp(-energy(rbm, v, h) for h in [[0], [1]])
    end

    rbm = BinaryRBM(randn(7), randn(2), randn(7, 2))
    v = bitrand(7)
    hs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    @test -free_energy(rbm, v) ≈ logsumexp(-energy(rbm, v, h) for h in hs)

    h = bitrand(2)
    @test @inferred(hidden_cgf(rbm, v)) ≈ cgf(rbm.hidden, inputs_h_from_v(rbm, v))
    @test @inferred(visible_cgf(rbm, h)) ≈ cgf(rbm.visible, inputs_v_from_h(rbm, h))
end

@testset "Gaussian-Gaussian RBM, 1-dimension" begin
    rbm = RBM(
        Gaussian(; θ = randn(1), γ = rand(1) .+ 1),
        Gaussian(; θ = randn(1), γ = rand(1) .+ 1),
        randn(1, 1) / 1.0e2
    )

    @assert isposdef(
        [
            rbm.visible.γ -rbm.w;
            -rbm.w'  rbm.hidden.γ
        ]
    )

    Z, ϵ = quadgk(x -> exp(-only(free_energy(rbm, [x;;]))), -Inf, Inf)
    @test log_partition(rbm) ≈ log(Z)
end

@testset "Gaussian-Gaussian RBM, multi-dimensional" begin
    n = (10, 3)
    m = (7, 2)
    rbm = RBM(
        Gaussian(; θ = randn(n...), γ = rand(n...) .+ 0.5),
        Gaussian(; θ = randn(m...), γ = rand(m...) .+ 0.5),
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
    A = [
        diagm(γv) -w;
        -w'  diagm(γh)
    ]

    v = randn(n..., 1)
    h = randn(m..., 1)
    x = [reshape(v, N, 1); reshape(h, M, 1)]

    @test energy(rbm, v, h) ≈ x' * A * x / 2 - θ' * x
    @test log_partition(rbm) ≈ (
        (N + M) / 2 * log(2π) + θ' * inv(A) * θ / 2 - logdet(A) / 2
    )
    @test log_likelihood(rbm, v) ≈ -free_energy(rbm, v) .- log_partition(rbm)

    Ev = sum(rbm.visible.γ .* v .^ 2 ./ 2 - rbm.visible.θ .* v)
    Fv = sum(-(rbm.hidden.θ .+ inputs_h_from_v(rbm, v)) .^ 2 ./ 2rbm.hidden.γ)
    @test only(free_energy(rbm, v)) ≈ Ev + Fv - sum(log.(2π ./ rbm.hidden.γ)) / 2

    gs = gradient(rbm) do rbm
        log_partition(rbm)
    end
    ∂θv = vec(only(gs).visible.par[1, ..])
    ∂γv = vec(only(gs).visible.par[2, ..])
    ∂θh = vec(only(gs).hidden.par[1, ..])
    ∂γh = vec(only(gs).hidden.par[2, ..])
    ∂w = reshape(only(gs).w, length(rbm.visible), length(rbm.hidden))

    Σ = inv(A) # covariances
    μ = Σ * θ  # means
    C = Σ + μ * μ' # non-centered second moments

    @test [∂θv; ∂θh] ≈ μ
    @test -2∂γv ≈ diag(C[1:N, 1:N]) # <vi^2>
    @test -2∂γh ≈ diag(C[(N + 1):end, (N + 1):end]) # <hμ^2>
    @test ∂w ≈ C[1:N, (N + 1):end] # <vi*hμ>
end

@testset "Gaussian-Gaussian RBM normalizability" begin
    for T in (Float32, Float64)
        I2 = diagm(ones(T, 2))
        θv = T[0.25, -0.5]
        θh = T[-0.75, 0.125]
        w = T(0.5) * I2

        rbm = GaussianRBM(θv, T[-1, 1], θh, T[1, -1], w)
        rbm_positive_γ = GaussianRBM(θv, ones(T, 2), θh, ones(T, 2), w)
        θ = [θv; θh]
        A = [I2 -w; -w' I2]
        expected = length(θ) / 2 * log(2π) + dot(θ, A \ θ) / 2 - logdet(A) / 2
        logZ = log_partition(rbm)

        @test logZ ≈ expected
        @test logZ ≈ log_partition(rbm_positive_γ)

        if T === Float64
            gs = only(gradient(log_partition, rbm))
            gs_positive_γ = only(gradient(log_partition, rbm_positive_γ))
            @test gs.visible.par[1, ..] ≈ gs_positive_γ.visible.par[1, ..]
            @test gs.hidden.par[1, ..] ≈ gs_positive_γ.hidden.par[1, ..]
            @test gs.visible.par[2, ..] ≈
                sign.(rbm.visible.γ) .* gs_positive_γ.visible.par[2, ..]
            @test gs.hidden.par[2, ..] ≈
                sign.(rbm.hidden.γ) .* gs_positive_γ.hidden.par[2, ..]
            @test gs.w ≈ gs_positive_γ.w
        end

        α = one(T) - sqrt(eps(T))
        δ = one(T) - α
        θ_boundary = fill(sqrt(δ), 1)
        rbm_boundary = GaussianRBM(
            θ_boundary, ones(T, 1),
            θ_boundary, ones(T, 1),
            reshape(T[α], 1, 1)
        )
        logZ_boundary = log_partition(rbm_boundary)
        expected_boundary = (
            log(T(2) * T(π)) + one(T) - (log(δ) + log1p(α)) / T(2)
        )
        @test isfinite(logZ_boundary)
        @test logZ_boundary ≈ expected_boundary rtol = sqrt(eps(T))

        w_indefinite = 2I2
        A_indefinite = [I2 -w_indefinite; -w_indefinite' I2]
        @test !isposdef(A_indefinite)
        @test logdet(A_indefinite) ≈ log(T(9))

        rbm_indefinite = GaussianRBM(
            zeros(T, 2), ones(T, 2), zeros(T, 2), ones(T, 2), w_indefinite
        )
        logZ_indefinite = log_partition(rbm_indefinite)
        @test isinf(logZ_indefinite) && logZ_indefinite > 0
        @test typeof(logZ_indefinite) === typeof(logZ)

        rbm_singular = GaussianRBM(
            zeros(T, 2), ones(T, 2), zeros(T, 2), ones(T, 2), I2
        )
        logZ_singular = log_partition(rbm_singular)
        @test isinf(logZ_singular) && logZ_singular > 0
        @test typeof(logZ_singular) === typeof(logZ)
    end
end

@testset "zero hidden units" begin
    rbm = BinaryRBM(randn(5), randn(0), randn(5, 0))
    v = bitrand(5)
    @test free_energy(rbm, v) ≈ energy(rbm.visible, v)
end

@testset "∂free_energy" begin
    rbm = BinaryRBM(randn(5, 2), randn(4, 3), randn(5, 2, 4, 3))
    v = bitrand(size(rbm.visible)..., 7)
    h = bitrand(size(rbm.hidden)..., 7)

    gs = gradient(rbm) do rbm
        mean(free_energy(rbm, v))
    end
    ∂F = @inferred ∂free_energy(rbm, v)
    ∂Fv = @inferred ∂free_energy_v(rbm, v)
    @test ∂F.visible ≈ ∂Fv.visible ≈ only(gs).visible.par
    @test ∂F.hidden ≈ ∂Fv.hidden ≈ only(gs).hidden.par
    @test ∂F.w ≈ ∂Fv.w ≈ only(gs).w

    gs = gradient(rbm) do rbm
        mean(free_energy_h(rbm, h))
    end
    ∂F = @inferred ∂free_energy_h(rbm, h)
    @test ∂F.visible ≈ only(gs).visible.par
    @test ∂F.hidden ≈ only(gs).hidden.par
    @test ∂F.w ≈ only(gs).w

    wts = rand(7)
    gs = gradient(rbm) do rbm
        wmean(free_energy(rbm, v); wts)
    end
    ∂F = ∂free_energy(rbm, v; wts)
    @test ∂F.visible ≈ only(gs).visible.par
    @test ∂F.hidden ≈ only(gs).hidden.par
    @test ∂F.w ≈ only(gs).w

    # ∂ algebra
    gλ = 2.3 * ∂F
    @test gλ.visible ≈ ∂F.visible * 2.3
    @test gλ.hidden ≈ ∂F.hidden * 2.3
    @test gλ.w ≈ ∂F.w * 2.3
end

@testset "total_mean_h_from_v, total_mean_v_from_h, etc." begin
    rbm = BinaryRBM(randn(5), randn(4), randn(5, 4))
    v = bitrand(size(rbm.visible)..., 10)
    h = bitrand(size(rbm.hidden)..., 10)

    @test dropdims(mean(mean_h_from_v(rbm, v); dims = 2); dims = 2) ≈ @inferred total_mean_h_from_v(rbm, v)
    @test dropdims(mean(mean_v_from_h(rbm, h); dims = 2); dims = 2) ≈ @inferred total_mean_v_from_h(rbm, h)

    @test dropdims(var(sample_h_from_v(rbm, repeat(v, 1, 1, 1000)); dims = (2, 3)); dims = (2, 3)) ≈ @inferred(total_var_h_from_v(rbm, v)) rtol = 0.1
    @test dropdims(var(sample_v_from_h(rbm, repeat(h, 1, 1, 1000)); dims = (2, 3)); dims = (2, 3)) ≈ @inferred(total_var_v_from_h(rbm, h)) rtol = 0.1

    μ, ν = total_meanvar_h_from_v(rbm, v)
    @test μ ≈ total_mean_h_from_v(rbm, v)
    @test ν ≈ total_var_h_from_v(rbm, v)

    μ, ν = total_meanvar_v_from_h(rbm, h)
    @test μ ≈ total_mean_v_from_h(rbm, h)
    @test ν ≈ total_var_v_from_h(rbm, h)
end

@testset "SpinRBM convenience constructors" begin
    rbm = SpinRBM(Float32, 5, 3)
    @test rbm.visible isa Spin
    @test rbm.hidden isa Spin
    @test size(rbm.w) == (5, 3)
    @test eltype(rbm.w) == Float32
    @test iszero(rbm.visible.θ)
    @test iszero(rbm.hidden.θ)
    @test iszero(rbm.w)

    rbm = SpinRBM(5, 3)
    @test rbm.visible isa Spin
    @test rbm.hidden isa Spin
    @test size(rbm.w) == (5, 3)
    @test eltype(rbm.w) == Float64
end

@testset "GaussianRBM convenience constructors" begin
    θv = randn(5)
    γv = rand(5) .+ 1
    θh = randn(3)
    γh = rand(3) .+ 1
    w = randn(5, 3)
    rbm = GaussianRBM(θv, γv, θh, γh, w)
    @test rbm.visible isa Gaussian
    @test rbm.hidden isa Gaussian
    @test rbm.visible.θ ≈ θv
    @test rbm.visible.γ ≈ γv
    @test rbm.hidden.θ ≈ θh
    @test rbm.hidden.γ ≈ γh
    @test rbm.w ≈ w

    rbm2 = GaussianRBM(θv, γv, w)
    @test rbm2.visible isa Gaussian
    @test rbm2.hidden isa Gaussian
    @test rbm2.visible.θ ≈ θv
    @test rbm2.visible.γ ≈ γv
    @test iszero(rbm2.hidden.θ)
    @test rbm2.hidden.γ == ones(3)
    @test rbm2.w ≈ w
end

@testset "HopfieldRBM type/dims constructors" begin
    rbm = HopfieldRBM(Float32, 5, 3)
    @test rbm.visible isa Spin
    @test rbm.hidden isa Gaussian
    @test size(rbm.w) == (5, 3)
    @test eltype(rbm.w) == Float32
    @test iszero(rbm.visible.θ)
    @test iszero(rbm.hidden.θ)
    @test rbm.hidden.γ == ones(Float32, 3)

    rbm = HopfieldRBM(5, 3)
    @test rbm.visible isa Spin
    @test rbm.hidden isa Gaussian
    @test size(rbm.w) == (5, 3)
    @test eltype(rbm.w) == Float64
end

@testset "reconstruction_error exact value for a deterministic RBM" begin
    # zero weights and huge positive fields: v -> h -> v' reconstruction is all ones,
    # so the error is the fraction of zeros in each sample
    rbm = BinaryRBM(fill(1.0e3, 3), fill(1.0e3, 2), zeros(3, 2))
    v = bitrand(3, 100)
    @test reconstruction_error(rbm, v) ≈ vec(mean(1 .- v; dims = 1))
    @test iszero(reconstruction_error(rbm, trues(3)))
    for steps in (1, 3)
        @test reconstruction_error(rbm, v; steps) ≈ vec(mean(1 .- v; dims = 1))
    end
end

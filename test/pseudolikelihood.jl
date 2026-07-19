using Test: @test, @testset
using Random: bitrand
using Statistics: mean, std
using LogExpFunctions: logsumexp
using RestrictedBoltzmannMachines: RBM, Binary, Gaussian, Spin, Potts, PottsGumbel, free_energy,
    substitution_matrix_exhaustive, log_pseudolikelihood, log_pseudolikelihood_sites,
    substitution_matrix_sites, log_pseudolikelihood_exact, onehot_encode

@testset "binary pseudolikelihood" begin
    n = (3, 2)
    m = (2,)
    B = 3

    rbm = RBM(Binary(n), Gaussian(m), randn(n..., m...) / √prod(n))
    v = bitrand(n..., B)

    E = free_energy(rbm, v)

    ΔE = zeros(2, n..., B)
    for (k, x) in enumerate((false, true)), i in CartesianIndices(n)
        v_ = copy(v)
        v_[i, :] .= x
        ΔE[k, i, :] .= free_energy(rbm, v_) - E
    end
    @test ΔE ≈ substitution_matrix_exhaustive(rbm, v)
    lpl = -logsumexp(-ΔE; dims = 1)
    @assert size(lpl) == (1, n..., B)
    @test log_pseudolikelihood(rbm, v; exact = true) ≈ vec(mean(lpl; dims = (1, 2, 3)))

    ΔE = zeros(2, B)
    sites = [rand(CartesianIndices(n)) for _ in 1:B]

    for (k, x) in enumerate((false, true))
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[i, b] = x
        end
        ΔE[k, :] .= free_energy(rbm, v_) - E
    end
    @test ΔE ≈ substitution_matrix_sites(rbm, v, sites)
    lpl = -logsumexp(-ΔE; dims = 1)
    @test log_pseudolikelihood_sites(rbm, v, sites) ≈ vec(lpl)
end

@testset "spins pseudolikelihood" begin
    n = (3, 2)
    m = (2,)
    B = 3

    rbm = RBM(Spin(n), Gaussian(m), randn(n..., m...) / √prod(n))
    v = rand((-1, +1), n..., B)

    E = free_energy(rbm, v)

    ΔE = zeros(2, n..., B)
    for (k, x) in enumerate((-1, 1)), i in CartesianIndices(n)
        v_ = copy(v)
        v_[i, :] .= x
        ΔE[k, i, :] .= free_energy(rbm, v_) - E
    end
    @test ΔE ≈ substitution_matrix_exhaustive(rbm, v)
    lpl = -logsumexp(-ΔE; dims = 1)
    @assert size(lpl) == (1, n..., B)
    @test log_pseudolikelihood(rbm, v; exact = true) ≈ vec(mean(lpl; dims = (1, 2, 3)))

    ΔE = zeros(2, B)
    sites = [rand(CartesianIndices(n)) for _ in 1:B]
    for (k, x) in enumerate((-1, 1))
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[i, b] = x
        end
        ΔE[k, :] .= free_energy(rbm, v_) - E
    end
    @test ΔE ≈ substitution_matrix_sites(rbm, v, sites)
    lpl = -logsumexp(-ΔE; dims = 1)
    @test log_pseudolikelihood_sites(rbm, v, sites) ≈ vec(lpl)
end

@testset "Potts pseudolikelihood" begin
    q = 3
    n = (3, 2)
    m = (2,)
    B = 3

    rbm = RBM(Potts((q, n...)), Gaussian(m), randn(q, n..., m...) / sqrt(q * prod(n)))
    v = falses(q, n..., B)
    for i in CartesianIndices(n), b in 1:B
        v[rand(1:q), i, b] = true
    end

    E = free_energy(rbm, v)

    ΔE = zeros(q, n..., B)
    for x in 1:q, i in CartesianIndices(n)
        v_ = copy(v)
        v_[:, i, :] .= false
        v_[x, i, :] .= true
        ΔE[x, i, :] .= free_energy(rbm, v_) - E
    end
    @test ΔE ≈ substitution_matrix_exhaustive(rbm, v)
    lpl = -logsumexp(-ΔE; dims = 1)
    @assert size(lpl) == (1, n..., B)
    @test log_pseudolikelihood(rbm, v; exact = true) ≈ vec(mean(lpl; dims = (1, 2, 3)))

    ΔE = zeros(q, B)
    sites = [rand(CartesianIndices(n)) for _ in 1:B]
    for x in 1:q
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[:, i, b] .= false
            v_[x, i, b] = true
        end
        ΔE[x, :] .= free_energy(rbm, v_) - E
    end
    @test ΔE ≈ substitution_matrix_sites(rbm, v, sites)
    lpl = -logsumexp(-ΔE; dims = 1)
    @test log_pseudolikelihood_sites(rbm, v, sites) ≈ vec(lpl)
end

# The specialized fast paths for Binary/Spin/Potts must agree with the generic
# reference implementation based on substitution matrices.
@testset "generic fallback vs fast path ($vis, $hid)" for
    vis in (:Binary, :Spin, :Potts), hid in (:Gaussian, :Binary)
    q = 3
    n = (3, 2)
    m = (2,)
    B = 4
    hidden = hid === :Gaussian ? Gaussian(m) : Binary(m)

    if vis === :Binary
        rbm = RBM(Binary(n), hidden, randn(n..., m...) / √prod(n))
        v = bitrand(n..., B)
    elseif vis === :Spin
        rbm = RBM(Spin(n), hidden, randn(n..., m...) / √prod(n))
        v = rand((-1, +1), n..., B)
    else
        rbm = RBM(Potts((q, n...)), hidden, randn(q, n..., m...) / sqrt(q * prod(n)))
        v = onehot_encode(rand(1:q, n..., B), 1:q)
    end
    sites = [rand(CartesianIndices(n)) for _ in 1:B]

    lpl_sites_generic = invoke(
        log_pseudolikelihood_sites,
        Tuple{RBM, AbstractArray, AbstractArray{<:CartesianIndex}},
        rbm, v, sites
    )
    @test lpl_sites_generic ≈ log_pseudolikelihood_sites(rbm, v, sites)

    lpl_exact_generic = invoke(log_pseudolikelihood_exact, Tuple{RBM, AbstractArray}, rbm, v)
    @test lpl_exact_generic ≈ log_pseudolikelihood_exact(rbm, v)
end

@testset "Gaussian-hidden exact fast paths" begin
    T = Float32
    n = (2, 2)
    m = (2, 2)
    batch = (2, 3)
    q = 3
    hidden = Gaussian(
        ; θ = randn(T, m...), γ = reshape(T[-2, -0.7, 1.1, 2.3], m)
    )
    v_potts = onehot_encode(rand(1:q, n..., batch...), 1:q)

    cases = (
        (Binary(; θ = randn(T, n...)), bitrand(n..., batch...)),
        (Spin(; θ = randn(T, n...)), rand((Int8(-1), Int8(1)), n..., batch...)),
        (Potts(; θ = randn(T, q, n...)), v_potts),
        (PottsGumbel(; θ = randn(T, q, n...)), v_potts),
    )

    for (visible, v) in cases
        w = randn(T, size(visible)..., m...) / sqrt(T(length(visible)))
        rbm = RBM(visible, hidden, w)
        reference = invoke(
            log_pseudolikelihood_exact, Tuple{RBM, AbstractArray}, rbm, v
        )
        result = log_pseudolikelihood_exact(rbm, v)
        @test size(result) == batch
        @test eltype(result) == T
        @test result ≈ reference rtol = 5.0e-4 atol = 5.0e-5
    end

    # The optimized Potts formula preserves the generic implementation's
    # behavior for dense site vectors, not only the usual one-hot inputs.
    visible = Potts(; θ = randn(T, q, n...))
    rbm = RBM(
        visible,
        hidden,
        randn(T, q, n..., m...) / sqrt(T(q * prod(n))),
    )
    v_dense = rand(T, q, n..., batch...)
    v_dense ./= sum(v_dense; dims = 1)
    reference = invoke(
        log_pseudolikelihood_exact, Tuple{RBM, AbstractArray}, rbm, v_dense
    )
    @test log_pseudolikelihood_exact(rbm, v_dense) ≈
        reference rtol = 5.0e-4 atol = 5.0e-5

    v_unbatched = bitrand(n...)
    rbm_unbatched = RBM(
        Binary(; θ = randn(T, n...)),
        hidden,
        randn(T, n..., m...) / sqrt(T(prod(n))),
    )
    result = log_pseudolikelihood_exact(rbm_unbatched, v_unbatched)
    reference = invoke(
        log_pseudolikelihood_exact,
        Tuple{RBM, AbstractArray},
        rbm_unbatched,
        v_unbatched,
    )
    @test size(result) == ()
    @test only(result) ≈ only(reference) rtol = 5.0e-4 atol = 5.0e-5

    # Normalize relative logits before logsumexp. Otherwise, a large common
    # term can erase the conditional normalization in finite precision.
    θ_offset = T[0, 1, -1]
    visible = Potts(; θ = reshape(θ_offset, q, 1))
    hidden_offset = Gaussian(; θ = T[1.0e8], γ = T[-1])
    rbm_offset = RBM(visible, hidden_offset, ones(T, q, 1, 1))
    for v in (
            onehot_encode(reshape([1], 1, 1), 1:q),
            reshape(T[0.2, 0.3, 0.5], q, 1, 1),
        )
        reference = invoke(
            log_pseudolikelihood_exact,
            Tuple{RBM{<:Potts}, AbstractArray},
            rbm_offset,
            v,
        )
        result = log_pseudolikelihood_exact(rbm_offset, v)
        expected = sum(v .* visible.θ) - logsumexp(θ_offset)
        @test only(reference) ≈ expected
        @test result ≈ reference
    end

    # Mixed-precision models retain visible fields that are finer than the
    # weights and hidden parameters.
    hidden_mixed = Gaussian(; θ = Float32[0], γ = Float32[1])
    mixed_cases = (
        (
            Binary(; θ = Float64[1.0e8 + 1]),
            reshape([false], 1),
            zeros(Float32, 1, 1),
        ),
        (
            Spin(; θ = Float64[5.0e7 + 0.5]),
            reshape(Int8[-1], 1),
            zeros(Float32, 1, 1),
        ),
        (
            Potts(; θ = reshape(Float64[1.0e8, 1.0e8 + 1, 1.0e8 - 1], q, 1)),
            reshape(Bool[1, 0, 0], q, 1),
            zeros(Float32, q, 1, 1),
        ),
        (
            PottsGumbel(; θ = reshape(Float64[1.0e8, 1.0e8 + 1, 1.0e8 - 1], q, 1)),
            reshape(Bool[1, 0, 0], q, 1),
            zeros(Float32, q, 1, 1),
        ),
    )
    for (visible_mixed, v, w) in mixed_cases
        rbm_mixed = RBM(visible_mixed, hidden_mixed, w)
        reference = invoke(
            log_pseudolikelihood_exact, Tuple{RBM, AbstractArray}, rbm_mixed, v
        )
        result = log_pseudolikelihood_exact(rbm_mixed, v)
        @test eltype(result) == Float64
        @test result ≈ reference rtol = 0 atol = 1.0e-8
    end

    # Integer parameters are valid even though inverse Gaussian precisions are
    # fractional.
    rbm_integer = RBM(
        Potts(; θ = zeros(Int, q, 1)),
        Gaussian(; θ = Int[0], γ = Int[2]),
        reshape(Int[0, 1, 2], q, 1, 1),
    )
    v_integer = reshape(Bool[1, 0, 0], q, 1)
    reference = invoke(
        log_pseudolikelihood_exact,
        Tuple{RBM{<:Potts}, AbstractArray},
        rbm_integer,
        v_integer,
    )
    result = log_pseudolikelihood_exact(rbm_integer, v_integer)
    @test eltype(result) == Float64
    @test result ≈ reference
end

@testset "stochastic log_pseudolikelihood" begin
    # With a single site, the stochastic estimator always selects that site,
    # so it coincides with the exact pseudolikelihood.
    rbm = RBM(Binary((1,)), Gaussian((2,)), randn(1, 2))
    v = bitrand(1, 7)
    @test log_pseudolikelihood(rbm, v) ≈ log_pseudolikelihood(rbm, v; exact = true)

    # With many sites, the estimator averages to the exact value. Repeat one
    # sample many times and bound the Monte Carlo error from the per-site spread.
    n = (3, 2)
    rbm = RBM(Binary(n), Gaussian((2,)), randn(n..., 2) / √prod(n))
    v0 = bitrand(n..., 1)
    exact = only(log_pseudolikelihood(rbm, v0; exact = true))
    persite = [only(log_pseudolikelihood_sites(rbm, v0, [i])) for i in CartesianIndices(n)]
    @test mean(persite) ≈ exact
    B = 2^12
    vrep = repeat(v0; outer = (1, 1, B))
    stoch = mean(log_pseudolikelihood(rbm, vrep))
    @test stoch ≈ exact atol = 10 * std(persite) / √B
end

@testset "substitution matrices without batch dimensions" begin
    n = (3, 2)
    rbm = RBM(Binary(n), Gaussian((2,)), randn(n..., 2) / √prod(n))
    v = bitrand(n...)
    F0 = free_energy(rbm, v)
    @test F0 isa Number

    site = fill(rand(CartesianIndices(n))) # 0-dimensional array of sites
    ΔE = substitution_matrix_sites(rbm, v, site)
    @test size(ΔE) == (2,)
    for (k, x) in enumerate((false, true))
        v_ = copy(v)
        v_[only(site)] = x
        @test ΔE[k] ≈ free_energy(rbm, v_) - F0
    end

    ΔE = substitution_matrix_exhaustive(rbm, v)
    @test size(ΔE) == (2, n...)
    for (k, x) in enumerate((false, true)), i in CartesianIndices(n)
        v_ = copy(v)
        v_[i] = x
        @test ΔE[k, i] ≈ free_energy(rbm, v_) - F0
    end
end

@testset "PottsGumbel pseudolikelihood delegates to Potts" begin
    q = 3
    n = (3, 2)
    m = (2,)
    B = 4

    rbm_potts = RBM(Potts((q, n...)), Gaussian(m), randn(q, n..., m...) / sqrt(q * prod(n)))
    rbm_gumbel = RBM(PottsGumbel(rbm_potts.visible), rbm_potts.hidden, rbm_potts.w)
    v = onehot_encode(rand(1:q, n..., B), 1:q)
    sites = [rand(CartesianIndices(n)) for _ in 1:B]

    @test substitution_matrix_sites(rbm_gumbel, v, sites) ≈ substitution_matrix_sites(rbm_potts, v, sites)
    @test substitution_matrix_exhaustive(rbm_gumbel, v) ≈ substitution_matrix_exhaustive(rbm_potts, v)
    @test log_pseudolikelihood_sites(rbm_gumbel, v, sites) ≈ log_pseudolikelihood_sites(rbm_potts, v, sites)
    @test log_pseudolikelihood(rbm_gumbel, v; exact = true) ≈ log_pseudolikelihood(rbm_potts, v; exact = true)
end

using Test: @test, @testset
using Random: bitrand
using Statistics: mean
using LogExpFunctions: logsumexp
using RestrictedBoltzmannMachines: RBM, Binary, Gaussian, Spin, Potts, free_energy,
    substitution_matrix_exhaustive, log_pseudolikelihood, log_pseudolikelihood_sites,
    substitution_matrix_sites

@testset "binary pseudolikelihood" begin
    n = (3,2)
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
    lpl = -logsumexp(-ΔE; dims=1)
    @assert size(lpl) == (1, n..., B)
    @test log_pseudolikelihood(rbm, v; exact=true) ≈ vec(mean(lpl; dims=(1,2,3)))

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
    lpl = -logsumexp(-ΔE; dims=1)
    @test log_pseudolikelihood_sites(rbm, v, sites) ≈ vec(lpl)
end

@testset "spins pseudolikelihood" begin
    n = (3,2)
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
    lpl = -logsumexp(-ΔE; dims=1)
    @assert size(lpl) == (1, n..., B)
    @test log_pseudolikelihood(rbm, v; exact=true) ≈ vec(mean(lpl; dims=(1,2,3)))

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
    lpl = -logsumexp(-ΔE; dims=1)
    @test log_pseudolikelihood_sites(rbm, v, sites) ≈ vec(lpl)
end

@testset "Potts pseudolikelihood" begin
    q = 3
    n = (3,2)
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
    lpl = -logsumexp(-ΔE; dims=1)
    @assert size(lpl) == (1, n..., B)
    @test log_pseudolikelihood(rbm, v; exact=true) ≈ vec(mean(lpl; dims=(1,2,3)))

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
    lpl = -logsumexp(-ΔE; dims=1)
    @test log_pseudolikelihood_sites(rbm, v, sites) ≈ vec(lpl)
end

using Test, Random, LinearAlgebra, Statistics, DelimitedFiles
import Zygote, Flux, Distributions, SpecialFunctions, LogExpFunctions, QuadGK, NPZ
import RestrictedBoltzmannMachines as RBMs

@testset "binary pseudolikelihood" begin
    n = (3,2)
    m = 2
    B = 3
    β = 1.5

    rbm = RBMs.RBM(RBMs.Binary(n...), RBMs.Gaussian(m...), randn(n..., m...) / √prod(n))
    v = bitrand(n..., B)

    E = RBMs.free_energy(rbm, v; β = β)

    ΔE = zeros(2, n..., B)
    for (k, x) in enumerate((false, true)), i in CartesianIndices(n)
        v_ = copy(v)
        v_[i, :] .= x
        ΔE[k, i, :] .= RBMs.free_energy(rbm, v_; β = β) - E
    end
    @test ΔE ≈ RBMs.substitution_matrix_exhaustive(rbm, v; β = β)
    lpl = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @assert size(lpl) == (1, n..., B)
    @test RBMs.log_pseudolikelihood(rbm, v; β = β, exact=true) ≈ vec(mean(lpl; dims=(1,2,3)))

    ΔE = zeros(2, B)
    sites = [rand(CartesianIndices(n)) for _ in 1:B]

    for (k, x) in enumerate((false, true))
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[i, b] = x
        end
        ΔE[k, :] .= RBMs.free_energy(rbm, v_; β = β) - E
    end
    @test ΔE ≈ RBMs.substitution_matrix_sites(rbm, v, sites; β = β)
    lpl = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @test RBMs.log_pseudolikelihood_sites(rbm, v, sites; β = β) ≈ vec(lpl)
end

@testset "spins pseudolikelihood" begin
    n = (3,2)
    m = 2
    B = 3
    β = 1.5

    rbm = RBMs.RBM(RBMs.Spin(n...), RBMs.Gaussian(m...), randn(n..., m...) / √prod(n))
    v = rand((-1, +1), n..., B)

    E = RBMs.free_energy(rbm, v; β = β)

    ΔE = zeros(2, n..., B)
    for (k, x) in enumerate((-1, 1)), i in CartesianIndices(n)
        v_ = copy(v)
        v_[i, :] .= x
        ΔE[k, i, :] .= RBMs.free_energy(rbm, v_; β = β) - E
    end
    @test ΔE ≈ RBMs.substitution_matrix_exhaustive(rbm, v; β = β)
    lpl = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @assert size(lpl) == (1, n..., B)
    @test RBMs.log_pseudolikelihood(rbm, v; β = β, exact=true) ≈ vec(mean(lpl; dims=(1,2,3)))

    ΔE = zeros(2, B)
    sites = [rand(CartesianIndices(n)) for _ in 1:B]
    for (k, x) in enumerate((-1, 1))
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[i, b] = x
        end
        ΔE[k, :] .= RBMs.free_energy(rbm, v_; β = β) - E
    end
    @test ΔE ≈ RBMs.substitution_matrix_sites(rbm, v, sites; β = β)
    lpl = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @test RBMs.log_pseudolikelihood_sites(rbm, v, sites; β = β) ≈ vec(lpl)
end

@testset "Potts pseudolikelihood" begin
    q = 3
    n = (3,2)
    m = 2
    B = 3
    β = 1.5

    rbm = RBMs.RBM(RBMs.Potts(q, n...), RBMs.Gaussian(m...), randn(q, n..., m...) / sqrt(q * prod(n)))
    v = falses(q, n..., B)
    for i in CartesianIndices(n), b in 1:B
        v[rand(1:q), i, b] = true
    end

    E = RBMs.free_energy(rbm, v; β = β)

    ΔE = zeros(q, n..., B)
    for x in 1:q, i in CartesianIndices(n)
        v_ = copy(v)
        v_[:, i, :] .= false
        v_[x, i, :] .= true
        ΔE[x, i, :] .= RBMs.free_energy(rbm, v_; β = β) - E
    end
    @test ΔE ≈ RBMs.substitution_matrix_exhaustive(rbm, v; β = β)
    lpl = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @assert size(lpl) == (1, n..., B)
    @test RBMs.log_pseudolikelihood(rbm, v; β = β, exact=true) ≈ vec(mean(lpl; dims=(1,2,3)))

    ΔE = zeros(q, B)
    sites = [rand(CartesianIndices(n)) for _ in 1:B]
    for x in 1:q
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[:, i, b] .= false
            v_[x, i, b] = true
        end
        ΔE[x, :] .= RBMs.free_energy(rbm, v_; β = β) - E
    end
    @test ΔE ≈ RBMs.substitution_matrix_sites(rbm, v, sites; β = β)
    lpl = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @test RBMs.log_pseudolikelihood_sites(rbm, v, sites; β = β) ≈ vec(lpl)
end

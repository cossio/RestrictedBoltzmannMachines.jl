using RestrictedBoltzmannMachines, Test, Random, Statistics
import StatsFuns: logaddexp
using RestrictedBoltzmannMachines: logsumexp, init_weights!

@testset "binary pseudolikelihood" begin
    n = (5,2)
    m = (4,3)
    B = (3,1)
    β = 2.0

    rbm = RBM(Binary(n...), Gaussian(m...))
    init_weights!(rbm)
    randn!(rbm.vis.θ)

    v = rand(Bool, n..., B...)
    xidx = siteindices(rbm.vis)
    bidx = batchindices(rbm.vis, v)
    sites = [rand(xidx) for b in bidx]
    lpl = log_pseudolikelihood(sites, rbm, v, β)
    lz = RBMs.log_site_traces(sites, rbm, v, β)
    @test size(lpl) == size(lz) == B

    for b in bidx
        site = sites[b]
        vb = v[xidx, b]
        @test size(vb) == size(rbm.vis)
        Fb = free_energy_v(rbm, vb, β)
        v_ = copy(vb)
        v_[site] = 1 - v_[site]
        F_ = free_energy_v(rbm, v_, β)
        @test lz[b] ≈ logaddexp(-β * Fb, -β * F_)
        @test lpl[b] ≈ -β * Fb - logaddexp(-β * Fb, -β * F_)
        @test lpl[b] ≈ log_pseudolikelihood(site, rbm, vb, β)
    end
end

@testset "spins pseudolikelihood" begin
    n = (5,2)
    m = (4,3)
    B = (3,1)
    β = 2.0

    rbm = RBM(Spin(n...), Gaussian(m...))
    init_weights!(rbm)
    randn!(rbm.vis.θ)
    I = randn(n..., B...)
    v = random(rbm.vis, I)

    xidx = siteindices(rbm.vis)
    bidx = batchindices(rbm.vis, v)
    sites = [rand(xidx) for b in bidx]
    lpl = log_pseudolikelihood(sites, rbm, v, β)
    lz = RBMs.log_site_traces(sites, rbm, v, β)
    @test size(lpl) == size(lz) == B

    for b in bidx
        site = sites[b]
        vb = v[xidx, b]
        @test size(vb) == size(rbm.vis)
        Fb = free_energy_v(rbm, vb, β)
        v_ = copy(vb)
        v_[site] = -v_[site]
        F_ = free_energy_v(rbm, v_, β)
        @test lz[b] ≈ logaddexp(-β * Fb, -β * F_)
        @test lpl[b] ≈ -β * Fb - logaddexp(-β * Fb, -β * F_)
        @test lpl[b] ≈ log_pseudolikelihood(site, rbm, vb, β)
    end
end

@testset "onehot pseudolikelihood" begin
    n = (3,5,2)
    m = (4,3)
    B = (3,2)
    β = 2.0

    rbm = RBM(Potts(n...), Gaussian(m...))
    init_weights!(rbm)
    randn!(rbm.vis.θ)

    v = random(rbm.vis, randn(n..., B...))
    xidx = siteindices(rbm.vis)
    bidx = batchindices(rbm.vis, v)
    sites = [rand(xidx) for b in bidx]
    lz = RBMs.log_site_traces(sites, rbm, v, β)
    lpl = log_pseudolikelihood(sites, rbm, v, β)
    @test size(lpl) == size(lz) == B

    for b in bidx
        vb = v[:, xidx, b]
        site = sites[b]
        v_ = copy(vb)
        v_[:, site] .= false
        F = zeros(rbm.vis.q)
        for a = 1:rbm.vis.q
            v_[a, site] = true
            F[a] = free_energy_v(rbm, v_, β)
            v_[a, site] = false
        end
        lZ = logsumexp(-β .* F)
        @test lz[b] ≈ lZ
        @test lpl[b] ≈ -β * free_energy_v(rbm, vb, β) - lZ
        @test lpl[b] ≈ log_pseudolikelihood(site, rbm, vb, β)
    end
end

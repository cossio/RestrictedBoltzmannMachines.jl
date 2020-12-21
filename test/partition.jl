using Test, Random, Statistics, LinearAlgebra
import StatsFuns: logaddexp
using Zygote
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: logsumexp, init_weights!

@testset "gaussian partition" begin
    rbm = RBM(Gaussian(10), Gaussian(7))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.vis.γ)
    rand!(rbm.hid.γ)
    rbm.weights .*= 1e-3
    A = [diagm(rbm.vis.γ) -rbm.weights;
         -rbm.weights' diagm(rbm.hid.γ)]
    v = randn(length(rbm.vis))
    h = randn(length(rbm.hid))
    @test energy(rbm, v, h) ≈ [v' h'] * A * [v; h] / 2 - rbm.vis.θ' * v - rbm.hid.θ' * h
    @test log_partition(rbm) ≈ -logdet(A)/2 + (length(rbm.vis) + length(rbm.hid)) / 2 * log(2π)
    @test log_partition(rbm, 1) ≈ -logdet(A)/2 + (length(rbm.vis) + length(rbm.hid)) / 2 * log(2π)
    @test log_partition(rbm, 2) ≈ -logdet(2A)/2 + (length(rbm.vis) + length(rbm.hid)) / 2 * log(2π)

    rbm.weights .= 0
    ps = params(rbm)
    gs = gradient(ps) do
        log_partition(rbm)
    end
    @test gs[rbm.vis.θ] ≈ mean(rbm.vis)
    @test gs[rbm.hid.θ] ≈ mean(rbm.hid)
end
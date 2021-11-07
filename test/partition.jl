using Test, Random, Statistics, LinearAlgebra
import StatsFuns: logaddexp
using Zygote, Flux
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: init_weights!

@testset "gaussian partition" begin
    rbm = RBM(Gaussian(10), Gaussian(7))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.visible.γ)
    rand!(rbm.hidden.γ)
    rbm.weights .*= 1e-3
    A = [diagm(rbm.visible.γ) -rbm.weights;
         -rbm.weights' diagm(rbm.hidden.γ)]
    v = randn(length(rbm.visible))
    h = randn(length(rbm.hidden))
    @test energy(rbm, v, h) ≈ [v' h'] * A * [v; h] / 2 - rbm.visible.θ' * v - rbm.hidden.θ' * h
    @test log_partition(rbm) ≈ -logdet(A)/2 + (length(rbm.visible) + length(rbm.hidden)) / 2 * log(2π)
    @test log_partition(rbm, 1) ≈ -logdet(A)/2 + (length(rbm.visible) + length(rbm.hidden)) / 2 * log(2π)
    @test log_partition(rbm, 2) ≈ -logdet(2A)/2 + (length(rbm.visible) + length(rbm.hidden)) / 2 * log(2π)

    rbm.weights .= 0
    ps = params(rbm)
    gs = gradient(ps) do
        log_partition(rbm)
    end
    @test gs[rbm.visible.θ] ≈ mean(rbm.visible)
    @test gs[rbm.hidden.θ] ≈ mean(rbm.hidden)
end

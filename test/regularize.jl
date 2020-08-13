using Test, Random, Statistics
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: fields_l2, weights_l1l2, l1l2, l2

@testset "regularization" begin
    w = randn(7,5)
    @test l1l2(w, Val(1)) ≈ mean(mean(abs.(w[:,μ]))^2 for μ=1:5)
    @inferred l1l2(w, Val(1))

    w = randn(4,2,3,5)
    @test l1l2(w, Val(2)) ≈ l1l2(reshape(w,8,15), Val(1))
    @inferred l1l2(w, Val(2))

    @test l2(w) ≈ mean(w.^2)
    @inferred l2(w)

    rbm = RBM(Binary(3,5), Gaussian(3,2))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    @test weights_l1l2(rbm) ≈ l1l2(rbm.weights, Val(2))
    @inferred weights_l1l2(rbm)
    @test fields_l2(rbm.vis) ≈ l2(rbm.vis.θ)
    @inferred fields_l2(rbm.vis)

    @test jerome_regularization(rbm; λv=0, λw=2) ≈ weights_l1l2(rbm)
    @test jerome_regularization(rbm; λv=2, λw=0) ≈ fields_l2(rbm.vis)
    @test jerome_regularization(rbm; λv=1, λw=1) ≈ jerome_regularization(rbm; λv=1, λw=0) + jerome_regularization(rbm; λv=0, λw=1)
    @inferred jerome_regularization(rbm; λv=1, λw=1)

    @test iszero(no_regularization(rbm))
    @inferred no_regularization(rbm)
end

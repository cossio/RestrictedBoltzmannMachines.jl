using Test: @test, @testset, @inferred
import Flux
import RestrictedBoltzmannMachines as RBMs
using Random: bitrand
using RestrictedBoltzmannMachines: default_optimizer, minibatch_count, minibatches,
    RBM, BinaryRBM, ∂free_energy, update!, ∂RBM,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU

@testset "default_optimizer" begin
    nsamples = 60000
    batchsize = 128
    epochs = 100
    decay_final = 0.01
    decay_after = 0.5
    steps_per_epoch = minibatch_count(nsamples; batchsize)
    nsteps = steps_per_epoch * epochs
    startepoch = round(Int, epochs * decay_after)

    o = default_optimizer(
        nsamples, batchsize, epochs;
        optim=Flux.Descent(1), decay_final, decay_after, clip=Inf,
    )
    p = [0.0]
    lr = [only(Flux.Optimise.apply!(o, p, [1.0])) for b in minibatches(nsamples; batchsize) for epoch in 1:epochs]

    decay_gamma = decay_final^(1 / (epochs - startepoch))
    lr_expected = [decay_gamma^max(n - startepoch, 0) for n in 1:epochs]
    lr_expected = repeat(lr_expected; inner=steps_per_epoch)
    # due to the way Flux.ExpDecay is defined, we must shift one step
    @test lr_expected[end] ≈ decay_final
    lr_expected = [lr_expected[2:end]; max(decay_final, decay_gamma * decay_final)]
    @test lr ≈ lr_expected

    @testset "no decay" begin
        decay_after = 2
        startepoch = round(Int, epochs * decay_after)

        o = default_optimizer(
            nsamples, batchsize, epochs;
            optim=Flux.Descent(1), decay_final, decay_after, clip=Inf,
        )
        p = [0.0]
        lr = [only(Flux.Optimise.apply!(o, p, [1.0])) for b in minibatches(nsamples; batchsize) for epoch in 1:epochs]
        @test all(lr .≈ 1)
    end
end

@testset "update!" begin
    opt = Flux.Descent(rand())
    rbm = RBM(Binary(; θ = randn(5)), Gaussian(; θ = randn(3), γ = randn(3)), randn(5,3))
    ∂ = ∂free_energy(rbm, bitrand(5, 10))
    Δ = deepcopy(∂)
    @test update!(Δ, rbm, opt) == Δ
    rbm0 = deepcopy(rbm)
    rbm = update!(rbm, Δ)
    @test rbm0.visible.par - rbm.visible.par ≈ Δ.visible ≈ ∂.visible * opt.eta
    @test rbm0.hidden.par - rbm.hidden.par ≈ Δ.hidden ≈ ∂.hidden * opt.eta
    @test rbm0.w - rbm.w ≈ Δ.w ≈ ∂.w * opt.eta
end

@testset "∂ operations" begin
    ∂1 = ∂RBM(randn(1,3), randn(1,2), randn(3,2))
    ∂2 = ∂RBM(randn(1,3), randn(1,2), randn(3,2))
    @test @inferred(∂1 + ∂2) == ∂RBM(∂1.visible + ∂2.visible, ∂1.hidden + ∂2.hidden, ∂1.w + ∂2.w)
    @test @inferred(∂1 - ∂2) == ∂RBM(∂1.visible - ∂2.visible, ∂1.hidden - ∂2.hidden, ∂1.w - ∂2.w)
    @test @inferred(2∂1) == ∂RBM(2∂1.visible, 2∂1.hidden, 2∂1.w)
end

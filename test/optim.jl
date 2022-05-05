using Test: @test, @testset, @inferred
import Flux
import RestrictedBoltzmannMachines as RBMs
using Random: bitrand
using RestrictedBoltzmannMachines: default_optimizer, minibatch_count, minibatches
using RestrictedBoltzmannMachines: RBM, BinaryRBM, ∂free_energy
using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU

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
end

@testset "update!" begin
    opt = Flux.Descent(rand())
    rbm = RBM(Binary(randn(5)), Spin(randn(3)), randn(5,3))
    ∂ = ∂free_energy(rbm, bitrand(5, 10))
    Δ = deepcopy(∂)
    @test RBMs.update!(Δ, rbm, opt) == Δ
    rbm0 = deepcopy(rbm)
    rbm = RBMs.update!(rbm, Δ)
    @test rbm0.visible.θ - rbm.visible.θ ≈ Δ.visible.θ ≈ ∂.visible.θ * opt.eta
    @test rbm0.hidden.θ - rbm.hidden.θ ≈ Δ.hidden.θ ≈ ∂.hidden.θ * opt.eta
    @test rbm0.w - rbm.w ≈ Δ.w ≈ ∂.w * opt.eta

    opt = Flux.Descent(rand())
    layer = Gaussian(randn(5), rand(5))
    ∂ = ∂free_energy(layer, randn(5, 10))
    Δ = deepcopy(∂)
    @test RBMs.update!(Δ, layer, opt) == Δ
    layer0 = deepcopy(layer)
    layer = RBMs.update!(layer, Δ)
    @test layer0.θ - layer.θ ≈ Δ.θ ≈ ∂.θ * opt.eta
    @test layer0.γ - layer.γ ≈ Δ.γ ≈ ∂.γ * opt.eta

    opt = Flux.Descent(rand())
    layer = dReLU(randn(5), randn(5), rand(5), rand(5))
    ∂ = ∂free_energy(layer, randn(5, 10))
    Δ = deepcopy(∂)
    @test RBMs.update!(Δ, layer, opt) == Δ
    layer0 = deepcopy(layer)
    layer = RBMs.update!(layer, Δ)
    @test layer0.θp - layer.θp ≈ Δ.θp ≈ ∂.θp * opt.eta
    @test layer0.θn - layer.θn ≈ Δ.θn ≈ ∂.θn * opt.eta
    @test layer0.γp - layer.γp ≈ Δ.γp ≈ ∂.γp * opt.eta
    @test layer0.γn - layer.γn ≈ Δ.γn ≈ ∂.γn * opt.eta

    opt = Flux.Descent(rand())
    layer = pReLU(randn(5), randn(5), rand(5), rand(5))
    ∂ = ∂free_energy(layer, randn(5, 10))
    Δ = deepcopy(∂)
    @test RBMs.update!(Δ, layer, opt) == Δ
    layer0 = deepcopy(layer)
    layer = RBMs.update!(layer, Δ)
    @test layer0.θ - layer.θ ≈ Δ.θ ≈ ∂.θ * opt.eta
    @test layer0.γ - layer.γ ≈ Δ.γ ≈ ∂.γ * opt.eta
    @test layer0.Δ - layer.Δ ≈ Δ.Δ ≈ ∂.Δ * opt.eta
    @test layer0.η - layer.η ≈ Δ.η ≈ ∂.η * opt.eta

    opt = Flux.Descent(rand())
    layer = xReLU(randn(5), randn(5), rand(5), rand(5))
    ∂ = ∂free_energy(layer, randn(5, 10))
    Δ = deepcopy(∂)
    @test RBMs.update!(Δ, layer, opt) == Δ
    layer0 = deepcopy(layer)
    layer = RBMs.update!(layer, Δ)
    @test layer0.θ - layer.θ ≈ Δ.θ ≈ ∂.θ * opt.eta
    @test layer0.γ - layer.γ ≈ Δ.γ ≈ ∂.γ * opt.eta
    @test layer0.Δ - layer.Δ ≈ Δ.Δ ≈ ∂.Δ * opt.eta
    @test layer0.ξ - layer.ξ ≈ Δ.ξ ≈ ∂.ξ * opt.eta
end


@testset "subtract_gradients" begin
    nt1 = (x = [2], y = [3])
    nt2 = (x = [1], y = [-1])
    @test @inferred(RBMs.subtract_gradients(nt1, nt2)) == (x = [1], y = [4])

    nt1 = (x = [2], y = [3], t = (a = [1], b = [2]))
    nt2 = (x = [1], y = [-1], t = (a = [2], b = [0]))
    @test @inferred(RBMs.subtract_gradients(nt1, nt2)) == (
        x = [1], y = [4], t = (a = [-1], b = [2])
    )
end

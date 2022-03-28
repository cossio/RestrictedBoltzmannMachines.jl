using Test: @test, @testset
import Flux
using RestrictedBoltzmannMachines: default_optimizer, minibatch_count, minibatches

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

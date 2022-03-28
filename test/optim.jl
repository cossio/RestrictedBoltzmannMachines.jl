using Test: @test, @testset
import Random
import Zygote
import Flux
import RestrictedBoltzmannMachines as RBMs

@testset "default_optimizer" begin
    nsamples = 60000
    batchsize = 128
    epochs = 100

    o = RBMs.default_optimizer(
        nsamples, batchsize, epochs;
        optim=Flux.Descent(1), decay_final=0.01, decay_after=0.5, clip=Inf,
    )
    p = [0.0]
    lrs = [only(Flux.Optimise.apply!(o, p, [1.0])) for b in RBMs.minibatches(nsamples; batchsize=batchsize) for epoch in 1:epochs]

    steps_per_epoch = RBMs.minibatch_count(nsamples; batchsize = batchsize)
    nsteps = steps_per_epoch * epochs
    start = round(Int, nsteps * 0.5)
    lrs_expected = [0.01^((max(n - start, 0) ÷ steps_per_epoch) / (max(nsteps - start, 0) ÷ steps_per_epoch)) for n in 1:nsteps]

    @test lrs ≈ lrs_expected
end

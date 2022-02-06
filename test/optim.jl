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

# @testset "SqrtDecay" begin
#     Random.seed!(84)
#     w = randn(10, 10)
#     w_ = randn(10, 10)
#     loss(x) = Flux.Losses.mse(w * x, w_ * x)
#     opt = Flux.Optimiser(RBMs.SqrtDecay(; decay=5), Flux.ADAM(0.001))
#     for t = 1:10^5
#         θ = Zygote.Params([w_])
#         x = rand(10)
#         θ_ = Zygote.gradient(() -> loss(x), θ)
#         Flux.Optimise.update!(opt, θ, θ_)
#     end
#     @test loss(rand(10, 10)) < 0.01
# end

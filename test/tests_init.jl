using Test, Random, LinearAlgebra, Statistics
using StatsFuns, Zygote, Flux, FiniteDifferences, OneHot, SpecialFunctions, Distributions
using LogExpFunctions: logaddexp, softmax, log1pexp, logsumexp, logistic
using Flux: params

using RestrictedBoltzmannMachines
import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: sum_, mean_, weighted_mean
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: inf, two, generate_sequences
using RestrictedBoltzmannMachines: tnmean, tnstd, tnvar, randnt, randnt_half, sqrt1half, relu_cgf
using RestrictedBoltzmannMachines: sum_, mean_, init_weights!, minibatches, minibatch_count

# using RestrictedBoltzmannMachines: sum_, mean_,
#     gauge, zerosum, rescale,
#     gauge!, zerosum!, rescale!,
#     gauge!, rescale!

function gradtest(f, args...)
    ftest(xs...) = sum(sin.(f(xs...)))
    us = @. 0.01 * randn(eltype(args), size(args))
    gs = gradient(ftest, args...)
    Δ = central_fdm(7,1)(ϵ -> ftest((args .+ ϵ .* us)...), 0)
    @test Δ ≈ sum(sum(u .* g) for (u,g) in zip(us, gs))
end

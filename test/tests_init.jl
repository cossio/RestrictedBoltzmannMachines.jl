using Test, Random, LinearAlgebra, Statistics
using StatsFuns, Zygote, Flux, FiniteDifferences, SpecialFunctions, Distributions
using LogExpFunctions: logaddexp, softmax, log1pexp, logsumexp, logistic
import Flux

using RestrictedBoltzmannMachines
import RestrictedBoltzmannMachines as RBMs

function gradtest(f, args...)
    ftest(xs...) = sum(sin.(f(xs...)))
    us = @. 0.01 * randn(eltype(args), size(args))
    gs = gradient(ftest, args...)
    Δ = central_fdm(7,1)(ϵ -> ftest((args .+ ϵ .* us)...), 0)
    @test Δ ≈ sum(sum(u .* g) for (u,g) in zip(us, gs))
end

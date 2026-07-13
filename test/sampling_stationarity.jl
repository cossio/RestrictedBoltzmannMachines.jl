#= Stationarity ("no drift") tests for the samplers.
https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/120

The model distribution p(v) ∝ exp(-free_energy(rbm, v)) must be *invariant* under the
Markov kernels used for sampling. We enumerate the visible space of a small machine,
draw the initial chains *exactly* from p(v), run the kernel, and assert the empirical
distribution stays at p(v) within Monte-Carlo error. Unlike "run long and compare
histograms", this is immune to mixing time: a chain started at the target must remain
at the target for *every* number of sweeps, so any drift beyond the Monte-Carlo floor
means the kernel does not leave p invariant, i.e. one of the conditionals is wrong.
Conversely, it cleanly separates "the sampler is broken" from "this RBM mixes slowly". =#

import Random
using Test: @test, @testset
using Statistics: mean
using LogExpFunctions: softmax
using StatsBase: sample, Weights
using EllipsisNotation: (..)
using QuadGK: quadgk
using RestrictedBoltzmannMachines: RBM, StandardizedRBM,
    Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU,
    free_energy, free_energy_h, energy, sample_v_from_v, sample_h_from_h,
    sample_h_from_v, mean_h_from_v, var_h_from_v, metropolis, zerosum, collect_states

Random.seed!(37)

enumerate_states(layer::Union{Binary,Spin}) = collect_states(layer)

# All one-hot configurations of a Potts/PottsGumbel layer, in a (size(layer)..., S) array
function enumerate_states(layer::Union{Potts,PottsGumbel})
    q = size(layer, 1)
    sites = length(layer) ÷ q
    states = falses(size(layer)..., q^sites)
    flat = reshape(states, q, sites, q^sites)
    for (n, colors) in enumerate(Iterators.product(ntuple(Returns(1:q), sites)...))
        for (site, color) in enumerate(colors)
            flat[color, site, n] = true
        end
    end
    return states
end

total_variation(p::AbstractVector, q::AbstractVector) = sum(abs, p - q) / 2

# Histogram of the chains `x` over the enumerated `states` (whose trailing dim indexes states)
function empirical_distribution(x::AbstractArray, states::AbstractArray)
    S = size(states)[end]
    index = Dict(vec(states[.., n]) => n for n in 1:S)
    counts = zeros(S)
    for b in 1:size(x)[end]
        counts[index[vec(x[.., b])]] += 1
    end
    return counts / size(x)[end]
end

#= Draws `nchains` chains exactly from p ∝ exp(logp) over `states`, and asserts that the
empirical distribution stays at p while running `kernel(x, steps)` for increasing numbers
of steps. The tolerance is a multiple of the Monte-Carlo floor of the (plug-in) total
variation estimate, so it scales correctly with `nchains` instead of being a magic
constant. Returns the final chains. =#
function check_no_drift(
    kernel, logp::AbstractVector, states::AbstractArray;
    nchains::Int = 20_000, steps = (1, 4, 16), nsigmas::Real = 4,
)
    p = softmax(logp)
    mcfloor = sum(sqrt.(p .* (1 .- p) ./ nchains)) / 2
    x = states[.., sample(1:length(p), Weights(p), nchains)]
    @test total_variation(empirical_distribution(x, states), p) < nsigmas * mcfloor # exact initial draw
    for k in steps
        x = kernel(x, k)
        @test total_variation(empirical_distribution(x, states), p) < nsigmas * mcfloor
    end
    return x
end

random_layer(::Type{Binary}, sz::Dims) = Binary(; θ = randn(sz...) / 2)
random_layer(::Type{Spin}, sz::Dims) = Spin(; θ = randn(sz...) / 2)
random_layer(::Type{Potts}, sz::Dims) = Potts(; θ = randn(sz...) / 2)
random_layer(::Type{PottsGumbel}, sz::Dims) = PottsGumbel(; θ = randn(sz...) / 2)
random_layer(::Type{Gaussian}, sz::Dims) = Gaussian(; θ = randn(sz...) / 2, γ = 0.5 .+ rand(sz...))
random_layer(::Type{ReLU}, sz::Dims) = ReLU(; θ = randn(sz...) / 2, γ = 0.5 .+ rand(sz...))
random_layer(::Type{dReLU}, sz::Dims) = dReLU(;
    θp = randn(sz...) / 2, θn = randn(sz...) / 2, γp = 0.5 .+ rand(sz...), γn = 0.5 .+ rand(sz...)
)
random_layer(::Type{pReLU}, sz::Dims) = pReLU(;
    θ = randn(sz...) / 2, γ = 0.5 .+ rand(sz...), Δ = randn(sz...) / 2, η = rand(sz...) .- 1/2
)
random_layer(::Type{xReLU}, sz::Dims) = xReLU(;
    θ = randn(sz...) / 2, γ = 0.5 .+ rand(sz...), Δ = randn(sz...) / 2, ξ = randn(sz...) / 2
)
random_layer(::Type{nsReLU}, sz::Dims) = nsReLU(;
    θ = randn(sz...) / 2, Δ = randn(sz...) / 2, ξ = randn(sz...) / 2
)

#= Couplings strong enough that a broken kernel drifts visibly: with weaker weights
(e.g. randn/3) a kernel that ignores the visible scales still equilibrates within the
4σ Monte-Carlo band below, whereas at this strength its drift exceeds it several-fold. =#
function random_rbm(V, vsz::Dims, H, M::Int; standardized::Bool)
    rbm = RBM(random_layer(V, vsz), random_layer(H, (M,)), randn(vsz..., M) * 0.6)
    standardized || return rbm
    # nontrivial offsets and scales, to exercise the standardized inputs_h_from_v / inputs_v_from_h
    return StandardizedRBM(
        rbm,
        rand(vsz...) / 3, randn(M) / 5,
        0.5 .+ rand(vsz...), 0.5 .+ rand(M),
    )
end

_visible = ((Binary, (5,)), (Spin, (5,)), (Potts, (3, 3)), (PottsGumbel, (3, 3)))
_hidden = (Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU)

# zerosum is only meaningful for Potts-family visible layers (a no-op otherwise)
_combos = [
    (V, vsz, H, standardized, zs)
    for (V, vsz) in _visible for H in _hidden for standardized in (false, true)
    for zs in (V <: Union{Potts, PottsGumbel} ? (false, true) : (false,))
]

@testset "no drift: $V × $H, standardized=$standardized, zerosum=$zs" for (V, vsz, H, standardized, zs) in _combos
    rbm = random_rbm(V, vsz, H, 2; standardized)
    if zs
        rbm = zerosum(rbm)
    end
    states = enumerate_states(rbm.visible)
    logp = -free_energy(rbm, states)
    v = check_no_drift((v, k) -> sample_v_from_v(rbm, v; steps = k), logp, states)

    #= The v-side conditional is exercised by the composite kernel above; check the h-side
    conditional sample_h_from_v against its exact first two moments, using the stationary
    chains as (a distribution of) conditioning states. =#
    h = sample_h_from_v(rbm, v)
    μ = mean_h_from_v(rbm, v)
    ν = var_h_from_v(rbm, v)
    B = size(v)[end]
    @test all(abs.(mean(h - μ; dims = 2)) .< 4 .* sqrt.(mean(ν; dims = 2) / B))
    @test mean(abs2, h - μ) ≈ mean(ν) rtol = 0.1
end

#= metropolis targets p_β(v) ∝ exp(-β free_energy(rbm, v)): Gibbs proposals accepted with
probability min(1, exp(-(β-1)ΔF)). At β = 1 every move is accepted (plain Gibbs). =#
@testset "no drift: metropolis, β=$β" for β in (0.5, 1.0, 2.0)
    rbm = random_rbm(Binary, (5,), dReLU, 2; standardized = false)
    states = enumerate_states(rbm.visible)
    logp = -β * free_energy(rbm, states)
    check_no_drift((v, k) -> metropolis(rbm, v; β, steps = k), logp, states)
end

# The mirrored statement for the hidden chains: p(h) ∝ exp(-free_energy_h(rbm, h))
@testset "no drift: sample_h_from_h, standardized=$standardized" for standardized in (false, true)
    rbm = RBM(random_layer(Gaussian, (3,)), random_layer(Binary, (3,)), randn(3, 3) * 0.6)
    if standardized
        rbm = StandardizedRBM(rbm, randn(3) / 3, rand(3) / 3, 0.5 .+ rand(3), 0.5 .+ rand(3))
    end
    states = enumerate_states(rbm.hidden)
    logp = -free_energy_h(rbm, states)
    check_no_drift((h, k) -> sample_h_from_h(rbm, h; steps = k), logp, states)
end

hidden_support(::ReLU) = (0.0, Inf)
hidden_support(::Union{Gaussian, dReLU, pReLU, xReLU, nsReLU}) = (-Inf, Inf)

#= Companion consistency check, essentially free once the enumeration exists: the
free energy must equal the exact marginalization -log ∫ dh exp(-energy(rbm, v, h)),
i.e. ∫ dh exp(free_energy(rbm, v) - energy(rbm, v, h)) = 1 for every visible state.
This pins the target distribution the stationarity tests above are checked against. =#
@testset "free_energy = -log ∫dh exp(-E): $V × $H, standardized=$standardized" for (V, vsz) in ((Binary, (3,)), (Potts, (3, 2))), H in _hidden, standardized in (false, true)
    rbm = random_rbm(V, vsz, H, 1; standardized)
    states = enumerate_states(rbm.visible)
    lo, hi = hidden_support(rbm.hidden)
    for n in 1:size(states)[end]
        v = states[.., n]
        F = free_energy(rbm, v)
        Z, ε = quadgk(h -> exp(F - energy(rbm, v, [h])), lo, hi)
        @test Z ≈ 1 rtol = 1e-6
    end
end

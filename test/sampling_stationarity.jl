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
using RestrictedBoltzmannMachines: RBM, StandardizedRBM, CenteredRBM,
    Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU,
    free_energy, free_energy_h, energy, sample_v_from_v, sample_h_from_h,
    sample_h_from_v, sample_from_inputs, mean_h_from_v, mean_v_from_h, var_h_from_v,
    metropolis, zerosum, collect_states

Random.seed!(37)

enumerate_states(layer::Union{Binary, Spin}) = collect_states(layer)

# All one-hot configurations of a Potts/PottsGumbel layer, in a (size(layer)..., S) array
function enumerate_states(layer::Union{Potts, PottsGumbel})
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
    θ = randn(sz...) / 2, γ = 0.5 .+ rand(sz...), Δ = randn(sz...) / 2, η = rand(sz...) .- 1 / 2
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
function random_rbm(V, vsz::Dims, H, hsz::Dims; standardized::Bool)
    rbm = RBM(random_layer(V, vsz), random_layer(H, hsz), randn(vsz..., hsz...) * 0.6)
    standardized || return rbm
    # nontrivial offsets and scales, to exercise the standardized inputs_h_from_v / inputs_v_from_h
    return StandardizedRBM(
        rbm,
        rand(vsz...) / 3, randn(hsz...) / 5,
        0.5 .+ rand(vsz...), 0.5 .+ rand(hsz...),
    )
end
random_rbm(V, vsz::Dims, H, M::Int; standardized::Bool) = random_rbm(V, vsz, H, (M,); standardized)

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
        @test Z ≈ 1 rtol = 1.0e-6
    end
end

#= Exact counterparts of the statistical tests above, free of Monte-Carlo error, for the
discrete layers. `sample_v_from_h` and `sample_h_from_v` draw the units of a discrete layer
independently, with probabilities set by the conditional means: P(x = 1) = μ for Binary,
P(s = ±1) = (1 ± μ) / 2 for Spin, and P(color) = μ[color] for the Potts layers. Hence
`state_probabilities` is the distribution over the states of a discrete layer from which
these functions sample (that their draws follow it is what the statistical tests check),
and the detailed balance of the Gibbs kernels becomes an identity between small matrices,
which must hold to rounding error. =#

const _DiscreteLayers = Union{Binary, Spin, Potts, PottsGumbel}

unit_probability(::Binary, μ::Real, x::Real) = x > 0 ? μ : 1 - μ
unit_probability(::Spin, μ::Real, x::Real) = (1 + x * μ) / 2
unit_probability(::Union{Potts, PottsGumbel}, μ::Real, x::Real) = x > 0 ? μ : one(μ) # one-hot

# P[n, b]: probability of the `n`-th of the `states` of the discrete `layer`, when its units
# have the conditional means `μ[.., b]`
function state_probabilities(layer::_DiscreteLayers, μ::AbstractArray, states::AbstractArray)
    X = reshape(states, length(layer), :)
    M = reshape(μ, length(layer), :)
    return [prod(unit_probability.(Ref(layer), M[:, b], X[:, n])) for n in axes(X, 2), b in axes(M, 2)]
end

function random_model(V, vsz::Dims, H, hsz::Dims, kind::Symbol)
    rbm = random_rbm(V, vsz, H, hsz; standardized = kind == :standardized)
    kind == :centered || return rbm
    return CenteredRBM(rbm, rand(vsz...) / 3, randn(hsz...) / 5)
end

_kinds = (:plain, :centered, :standardized)
_discrete = ((Binary, (3,)), (Spin, (3,)), (Potts, (3, 2)), (PottsGumbel, (2, 2)))
_continuous = ((Gaussian, (2,)), (ReLU, (2,)), (dReLU, (2,)), (pReLU, (2,)), (xReLU, (2,)), (nsReLU, (2,)))

#= The distribution that a discrete layer is sampled from, given a configuration of the
other layer, is the conditional of the joint distribution p(v, h) ∝ exp(-energy(rbm, v, h)):
the Boltzmann weights of its states, normalized. The other layer can be of any type. =#
@testset "conditionals of the joint distribution: $V × $H, $kind" for (V, vsz) in (_discrete..., (Gaussian, (3,))), (H, hsz) in (_discrete..., _continuous...), kind in _kinds
    rbm = random_model(V, vsz, H, hsz, kind)
    B = 5
    v = sample_from_inputs(rbm.visible, zeros(vsz..., B))
    h = sample_from_inputs(rbm.hidden, zeros(hsz..., B))
    if rbm.visible isa _DiscreteLayers
        states = enumerate_states(rbm.visible)
        E = stack(energy(rbm, states, h[.., b]) for b in 1:B) # states × batch
        @test state_probabilities(rbm.visible, mean_v_from_h(rbm, h), states) ≈ softmax(-E; dims = 1) rtol = 1.0e-10
    end
    if rbm.hidden isa _DiscreteLayers
        states = enumerate_states(rbm.hidden)
        E = stack(energy(rbm, v[.., b], states) for b in 1:B)
        @test state_probabilities(rbm.hidden, mean_h_from_v(rbm, v), states) ≈ softmax(-E; dims = 1) rtol = 1.0e-10
    end
end

#= With both layers discrete, the transition matrices of sample_v_from_v and sample_h_from_h
are explicit: K(v → v') = Σ_h P(v' | h) P(h | v). They must satisfy detailed balance,
π(v) K(v → v') = π(v') K(v' → v), with respect to π(v) ∝ exp(-free_energy(rbm, v)), which
implies that π is stationary. =#
@testset "detailed balance of the Gibbs kernels: $V × $H, $kind" for (V, vsz) in _discrete, (H, hsz) in _discrete, kind in _kinds
    rbm = random_model(V, vsz, H, hsz, kind)
    vs = enumerate_states(rbm.visible)
    hs = enumerate_states(rbm.hidden)
    Phv = state_probabilities(rbm.hidden, mean_h_from_v(rbm, vs), hs) # P(h | v), h × v
    Pvh = state_probabilities(rbm.visible, mean_v_from_h(rbm, hs), vs) # P(v | h), v × h
    πv = softmax(-free_energy(rbm, vs))
    πh = softmax(-free_energy_h(rbm, hs))
    @test all(≈(1), sum(Phv; dims = 1))
    @test all(≈(1), sum(Pvh; dims = 1))

    # both factorizations of the joint distribution agree with its Boltzmann weights
    E = stack(energy(rbm, vs, hs[.., n]) for n in 1:size(hs)[end]) # v × h
    joint = reshape(softmax(-vec(E)), size(E))
    @test Pvh .* πh' ≈ joint rtol = 1.0e-10
    @test (Phv .* πv')' ≈ joint rtol = 1.0e-10

    for (K, π) in ((Pvh * Phv, πv), (Phv * Pvh, πh))
        flux = K .* π' # flux[x', x] = π(x) K(x → x')
        @test flux ≈ flux' rtol = 1.0e-10
        @test K * π ≈ π rtol = 1.0e-10
    end
end

#= The largest gap between the empirical distribution of `samples` and the distribution of
density `pdf` (supported on x ≥ lo), over a grid spanning the samples: a lower bound of the
Kolmogorov–Smirnov statistic. =#
function ks_distance(samples::AbstractVector, pdf, lo::Real; ngrid::Int = 400)
    x = sort(samples)
    D = cdf = 0.0
    a = float(lo)
    for b in range(first(x), last(x), ngrid)
        cdf += first(a < 0 < b ? quadgk(pdf, a, 0, b) : quadgk(pdf, a, b)) # the density can have a kink at 0
        D = max(D, abs(searchsortedlast(x, b) / length(x) - cdf))
        a = b
    end
    return D
end

#= A continuous hidden unit cannot be enumerated, but sample_h_from_v must draw it from the
density p(h | v) = exp(free_energy(rbm, v) - energy(rbm, v, h)), the conditional of the joint
distribution (normalized, by the marginalization test above). The moments checked above see
only the first two cumulants; a Kolmogorov–Smirnov test compares the whole distribution. For
n exact samples, the statistic exceeds c / √n with probability at most 2exp(-2c²), which is
below 1e-5 for c = 2.5. =#
@testset "sample_h_from_v draws from exp(F(v) - E(v, h)): $H, $kind" for H in _hidden, kind in _kinds
    rbm = random_model(Binary, (3,), H, (1,), kind)
    v = rand(Bool, 3)
    F = free_energy(rbm, v)
    n = 10^6
    h = vec(sample_h_from_v(rbm, repeat(v, 1, n)))
    @test ks_distance(h, x -> exp(F - energy(rbm, v, [x])), first(hidden_support(rbm.hidden))) < 2.5 / sqrt(n)
end

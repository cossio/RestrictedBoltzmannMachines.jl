#=
Rejection sampler based on algorithm from Robert (1995)

  - Available at http://arxiv.org/abs/0907.4010

Implementation from Distributions.jl, with few modifications to make it more generic.
Also here we avoid the overhead of computing the CDF, which `truncated(Normal(), a, b)`
always does when it constructs the truncated distribution (`tp`).
We specialize to the case where `b = Inf`.
=#

"""
    randnt([rng], a)

Random standard normal lower truncated at `a` (that is, Z ≥ a).
"""
randnt(rng::AbstractRNG, a::Real) = randnt(rng, float(a))
randnt(rng::AbstractRNG, a::BigFloat) = randnt(rng, Float64(a))
randnt(a::Real) = randnt(default_rng(), a)

# This union mirrors private `Base.IEEEFloat` without relying on a non-public name.
function randnt(rng::AbstractRNG, a::Union{Float16, Float32, Float64})
    local r = a
    if a ≤ 0
        while true
            r = randn(rng, typeof(a))
            r ≥ a && break
        end
    elseif a < Inf
        t = sqrt1half(a)
        while true
            r = a + randexp(rng, typeof(a)) / t
            u = rand(rng, typeof(a))
            u < exp(-(r - t)^2 / 2) && break
        end
    end
    return r
end

"""
    sqrt1half(x)

Accurate computation of sqrt(1 + (x/2)^2) + |x|/2.
"""
function sqrt1half(x::Real)
    h = x / 2
    return hypot(one(h), h) + abs(h)
end

"""
    randnt_half([rng], μ, σ)

Samples the normal distribution with mean `μ` and standard deviation `σ`
truncated to positive values.
"""
function randnt_half(rng::AbstractRNG, μ::Real, σ::Real)
    z = randnt(rng, -μ / σ)
    return μ + σ * z
end

randnt_half(μ::Real, σ::Real) = randnt_half(default_rng(), μ, σ)

"""
    tnmean(a)

Mean of the standard normal distribution,
truncated to the interval (a, +∞).
"""
function tnmean(a::Real)
    two = oftype(a, 2)
    return sqrt(two / π) / erfcx(a / sqrt(two))
end

"""
    tnvar(a)

Variance of the standard normal distribution,
truncated to the interval (a, +∞).
"""
tnvar(a::Real) = last(tnmeanvar(a))

"""
    tnmeanvar(a)

Mean and variance of the standard normal distribution truncated to the interval (a, +∞).
Equivalent to `tnmean(a), tnvar(a)` but saves some common computations.
"""
function tnmeanvar(a::Real)
    if a ≥ 8
        δ, ν = _tnmeanvar_tail(a)
        return a + δ, ν
    end
    μ = tnmean(a)
    ν = one(μ) - (μ - a) * μ
    return μ, ν
end

# broadcasts and returns a tuple of arrays, rather than an array of tuples like tnmeanvar.(A)
tnmeanvar(a::AbstractArray) = _unzip(tnmeanvar.(a))

# For a ≥ 8, compute the excess mean δ = E[Z | Z > a] - a directly from
# Laplace's continued fraction δ = 1 / (a + 2 / (a + 3 / (a + ...))).
# The variance is -δ'(a); differentiate the backward recurrence alongside δ
# instead of subtracting nearly equal numbers in 1 - (a + δ)δ.
function _tnmeanvar_tail(a::Real)
    δ = zero(float(a))
    ν = zero(δ)
    # 32 levels suffice for Float64 at a = 8 (relative error < 4e-25).
    # The required depth grows quadratically with precision for BigFloat.
    depth = max(32, cld(precision(δ)^2, 256))
    for k in depth:-1:1
        t = inv(a + δ)
        δ = k * t
        ν = δ * t * (1 - ν)
    end
    return δ, ν
end

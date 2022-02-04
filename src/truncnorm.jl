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
randnt(rng::Random.AbstractRNG, a::Real) = randnt(rng, float(a))
randnt(rng::Random.AbstractRNG, a::BigFloat) = randnt(rng, Float64(a))
randnt(a::Real) = randnt(Random.GLOBAL_RNG, a)

function randnt(rng::Random.AbstractRNG, a::Base.IEEEFloat)
    if a ≤ 0
        while true
            r = randn(rng, typeof(a))
            r ≥ a && return r
        end
    else
        t = sqrt1half(a)
        !(t < Inf) && return a
        while true
            r = a + Random.randexp(rng, typeof(a)) / t
            u = rand(rng, typeof(a))
            if u < exp(-(r - t)^2 / 2)
                return r
            end
        end
    end
end

"""
    sqrt1half(x)

Accurate computation of sqrt(1 + (x/2)^2) + |x|/2.
"""
sqrt1half(x::Real) = _sqrt1half(float(abs(x)))

function _sqrt1half(x::Real)
    if x > 2/sqrt(eps(x))
        return x
    else
        return sqrt(one(x) + (x/2)^2) + x/2
    end
end

"""
    randnt_half([rng], μ, σ)

Samples the normal distribution with mean `μ` and standard deviation `σ`
truncated to positive values.
"""
function randnt_half(rng::Random.AbstractRNG, μ::Real, σ::Real)
    z = randnt(rng, -μ / σ)
    return μ + σ * z
end

randnt_half(μ::Real, σ::Real) = randnt_half(Random.GLOBAL_RNG, μ, σ)

"""
    tnmean(a)

Mean of the standard normal distribution,
truncated to the interval (a, +∞).
"""
tnmean(a::Real) = sqrt(two(a)/π) / SpecialFunctions.erfcx(a/√two(a))

"""
    tnvar(a)

Variance of the standard normal distribution,
truncated to the interval (a, +∞).
WARNING: Fails for very very large values of `a`.
"""
function tnvar(a::Real)
    μ = tnmean(a)
    return one(μ) - (μ - a) * μ
end

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
function randnt(rng::AbstractRNG, a::Real)
    T = typeof(float(a))
    if a ≤ 0
        while true
            r = randn(rng, T)
            r ≥ a && return r
        end
    else
        t = sqrt1half(a)
        !(t < Inf) && return float(a)
        while true
            r = a + randexp(rng, T) / t
            u = rand(rng, T)
            if u < exp(-(r - t)^2 / 2)
                return r
            end
        end
    end
end
randnt(a::Real) = randnt(Random.GLOBAL_RNG, a)

"""
    sqrt1half(x)

Accurate computation of sqrt(1 + (x/2)^2) + |x|/2.
"""
sqrt1half(x::Real) = _sqrt1half(float(abs(x)))
function _sqrt1half(x)
    if x > 2/sqrt(eps(x))
        return x
    else
        return sqrt(one(x) + (x/2)^2) + x/2
    end
end

#=
Compute gradients using the approach of:
http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients
=#

@adjoint function randnt(rng::AbstractRNG, a::Real)
    z, da = ∇randnt(rng, a)
    return z, δ -> (nothing, δ * da)
end
@adjoint function broadcasted(::typeof(randnt), rng::AbstractRNG, a::Num)
    z, da = ∇randnt(rng, a)
    return z, δ -> (nothing, nothing, δ .* da)
end
function ∇randnt(rng::AbstractRNG, a::Num)
    z = randnt.(rng, a)
    log1mu = @. logerfc(z / √two(z)) - logerfc(a / √two(a))
    da = @. exp((z - a) * (z + a) / 2 + log1mu)
    return z, da
end

"""
    randnt_half([rng], μ, σ)

Samples the normal distribution with mean `μ` and standard deviation `σ`
truncated to non-negative values.
"""
function randnt_half(rng::AbstractRNG, μ::Real, σ::Real)
    z = randnt(rng, -μ / σ)
    return μ + σ * z
end
randnt_half(μ::Real, σ::Real) = randnt_half(GLOBAL_RNG, μ, σ)
@adjoint function randnt_half(rng::AbstractRNG, μ::Real, σ::Real)
    z, dμ, dσ = ∇randnt_half(rng, μ, σ)
    return z, δ -> (nothing, δ * dμ, δ * dσ)
end
@adjoint function broadcasted(::typeof(randnt_half), rng::AbstractRNG, μ::Num, σ::Num)
    z, dμ, dσ = ∇randnt_half(rng, μ, σ)
    back(δ) = (nothing, nothing, δ .* dμ, δ .* dσ)
    return z, δ -> (nothing, nothing, δ .* dμ, δ .* dσ)
end
function ∇randnt_half(rng::AbstractRNG, μ::Num, σ::Num)
    α = @. -μ / σ
    ζ, dα = ∇randnt(rng, α)
    z = @. μ + σ * ζ
    dμ = @. 1 - dα
    dσ = @. ζ - dα * α
    return z, dμ, dσ
end

@adjoint function randnt_half(μ::Real, σ::Real)
    z, dμ, dσ = ∇randnt_half(μ, σ)
    return z, δ -> (δ * dμ, δ * dσ)
end
@adjoint function broadcasted(::typeof(randnt_half), μ::Num, σ::Num)
    z, dμ, dσ = ∇randnt_half(μ, σ)
    return z, δ -> (nothing, δ .* dμ, δ .* dσ)
end
∇randnt_half(μ::Num, σ::Num) = ∇randnt_half(GLOBAL_RNG, μ, σ)

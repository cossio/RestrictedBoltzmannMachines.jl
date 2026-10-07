"""
    CossimDescent(η = 1e-3, ηmax = 1e-1, δ = 2e-3)

Gradient descent whose learning rate adapts to the alignment of successive gradients,
as used for [`ptt!`](@ref). The learning rate is multiplied by `1 + δ` when the cosine
similarity between the current and previous gradient is positive (capped at `ηmax`), and by
`1 - δ` when it is negative. Each parameter array adapts its own learning rate, starting
from `η`.

The rule assumes that the minibatches are drawn independently, as [`ptt!`](@ref) and
[`pcd!`](@ref) do. The minibatches of an epoch, drawn without replacement, have
anticorrelated noise, which makes successive gradients anti-aligned more often than not
once the noise dominates: with few minibatches per epoch, the learning rate then decays
geometrically.
"""
struct CossimDescent{T <: Real} <: AbstractRule
    eta::T
    etamax::T
    delta::T
    function CossimDescent(η::Real = 1.0e-3, ηmax::Real = 1.0e-1, δ::Real = 2.0e-3)
        0 < η ≤ ηmax || throw(ArgumentError("expected 0 < η ≤ ηmax, got η = $η, ηmax = $ηmax"))
        0 ≤ δ < 1 || throw(ArgumentError("expected 0 ≤ δ < 1, got δ = $δ"))
        T = float(promote_type(typeof(η), typeof(ηmax), typeof(δ)))
        return new{T}(η, ηmax, δ)
    end
end

# state: the previous gradient and the current learning rate
Optimisers.init(o::CossimDescent, x::AbstractArray) = (zero(x), o.eta)

function Optimisers.apply!(o::CossimDescent, (g, η), x::AbstractArray, dx::AbstractArray)
    c = dot(dx, g) # same sign as the cosine similarity
    if c > 0
        η = min(η * (1 + o.delta), o.etamax)
    elseif c < 0
        η = η * (1 - o.delta)
    end
    copyto!(g, dx)
    return (g, η), dx .* convert(float(real(eltype(x))), η)
end

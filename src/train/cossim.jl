"""
    CossimDescent(η = 1e-3, ηmax = 1e-1, δ = 2e-3)

Gradient descent whose learning rate adapts to the alignment of successive gradients,
as used for [`ptt!`](@ref). The learning rate is multiplied by `1 + δ` when the cosine
similarity between the current and previous gradient is positive (capped at `ηmax`), and by
`1 - δ` when it is negative. Each parameter array adapts its own learning rate, starting
from `η`.
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

# Multiplies the learning rate of every parameter in the optimiser state tree by `factor`.
_scale_learning_rate!(tree::Union{Tuple, NamedTuple}, factor::Real) =
    foreach(leaf -> _scale_learning_rate!(leaf, factor), tree)
function _scale_learning_rate!(leaf, factor::Real) # an `Optimisers.Leaf`
    leaf.rule, leaf.state = _scale_learning_rate(leaf.rule, leaf.state, factor)
    return nothing
end
_scale_learning_rate(o::CossimDescent, (g, η), factor::Real) = o, (g, min(η * factor, o.etamax))
function _scale_learning_rate(o::AbstractRule, state, factor::Real)
    hasproperty(o, :eta) || throw(ArgumentError("cannot scale the learning rate of $o"))
    return Optimisers.adjust(o, o.eta * factor), state
end

#= Discards the memory of past gradients (momenta, moment estimates) in the optimiser state
tree of the parameters `ps`, keeping the learning rates. =#
_reset_optimiser!(tree::Union{Tuple, NamedTuple}, ps) = foreach(_reset_optimiser!, tree, ps)
_reset_optimiser!(::Tuple{}, ps) = nothing # parameters without optimiser state
function _reset_optimiser!(leaf, x) # an `Optimisers.Leaf` of the parameters `x`
    leaf.state = _reset_optimiser(leaf.rule, leaf.state, x)
    return nothing
end
_reset_optimiser(::CossimDescent, (g, η), x::AbstractArray) = (zero(g), η)
_reset_optimiser(o::AbstractRule, state, x::AbstractArray) = Optimisers.init(o, x)

"""
    AbstractRegularizer

Supertype of the regularizers: penalties added to the training objective of an RBM. A
regularizer `reg` implements two methods for a plain `RBM`:

- `∂regularize!(∂, rbm, reg)`, which adds the gradient of its penalty to the gradient `∂`
  (a `∂RBM`, such as returned by [`∂free_energy`](@ref)) and returns `∂`;
- `regularization_penalty(rbm, reg)`, the value of its penalty.

These also regularize a `StandardizedRBM`: the penalty applies to the parameters of its
equivalent plain `RBM` ([`unstandardize`](@ref)), and its gradient is pulled back to the
parameters of the `StandardizedRBM`, with the offsets and scales held fixed. To penalize
the standardized parameters themselves instead, wrap the regularizer in a
[`StandardizedParametersRegularizer`](@ref). Several regularizers are combined by a
[`CompositeRegularizer`](@ref).
"""
abstract type AbstractRegularizer end

"""
    L2FieldsRegularizer(λ)

L2 penalty `λ/2 * sum(abs2, θ)` on the fields `θ` of the visible layer (`θp` and `θn` for
a `dReLU` layer).
"""
struct L2FieldsRegularizer{T <: Real} <: AbstractRegularizer
    λ::T
end

"""
    L1WeightsRegularizer(λ)

L1 penalty `λ * sum(abs, w)` on the weights.
"""
struct L1WeightsRegularizer{T <: Real} <: AbstractRegularizer
    λ::T
end

"""
    L2WeightsRegularizer(λ)

L2 penalty `λ/2 * sum(abs2, w)` on the weights.
"""
struct L2WeightsRegularizer{T <: Real} <: AbstractRegularizer
    λ::T
end

"""
    L2L1WeightsRegularizer(λ)

Penalty `λ/(2N) * sum(abs2, sum(abs, w; dims = visible))` on the weights, where `N` is the
number of visible units and the inner sum runs over the weights of each hidden unit (the
``L_2^1`` penalty of [Tubiana et al., 2019](https://doi.org/10.7554/eLife.39397), Eq. 8).
It promotes sparse weights, penalizing each hidden unit in proportion to its weights.
"""
struct L2L1WeightsRegularizer{T <: Real} <: AbstractRegularizer
    λ::T
end

"""
    CompositeRegularizer(regularizers...)

The sum of the given regularizers. Without arguments, no regularization.
"""
struct CompositeRegularizer{T <: Tuple{Vararg{AbstractRegularizer}}} <: AbstractRegularizer
    regularizers::T
end

CompositeRegularizer(regularizers::AbstractRegularizer...) = CompositeRegularizer(regularizers)

"""
    StandardizedParametersRegularizer(regularizer)

On a `StandardizedRBM`, applies `regularizer` to the standardized parameters themselves
(its layers and weights as stored) instead of to the parameters of the equivalent plain
`RBM`. On a plain `RBM` it is the same as `regularizer`.
"""
struct StandardizedParametersRegularizer{R <: AbstractRegularizer} <: AbstractRegularizer
    regularizer::R
end

"""
    ∂regularize!(∂, rbm, regularizer)

Adds the gradient of the penalty of `regularizer` (an [`AbstractRegularizer`](@ref)) with
respect to the parameters of `rbm` to the gradient `∂` (a `∂RBM`, such as returned by
[`∂free_energy`](@ref)), and returns `∂`. On a `StandardizedRBM` the penalty applies to the
parameters of the equivalent plain `RBM` ([`unstandardize`](@ref)), and its gradient is
pulled back to the parameters of `rbm`, with the offsets and scales held fixed, unless the
regularizer is a [`StandardizedParametersRegularizer`](@ref).
"""
function ∂regularize!(∂::∂RBM, rbm::RBM, reg::L2FieldsRegularizer)
    ∂regularize_fields!(∂.visible, rbm.visible, reg.λ)
    return ∂
end

function ∂regularize!(∂::∂RBM, rbm::RBM, reg::L1WeightsRegularizer)
    ∂.w .+= reg.λ .* sign.(rbm.w)
    return ∂
end

function ∂regularize!(∂::∂RBM, rbm::RBM, reg::L2WeightsRegularizer)
    ∂.w .+= reg.λ .* rbm.w
    return ∂
end

function ∂regularize!(∂::∂RBM, rbm::RBM, reg::L2L1WeightsRegularizer)
    dims = ntuple(identity, ndims(rbm.visible))
    ∂.w .+= reg.λ .* sign.(rbm.w) .* mean(abs, rbm.w; dims)
    return ∂
end

∂regularize!(∂::∂RBM, rbm::RBM, reg::CompositeRegularizer) = _∂regularize_composite!(∂, rbm, reg)
∂regularize!(∂::∂RBM, rbm::RBM, reg::StandardizedParametersRegularizer) = ∂regularize!(∂, rbm, reg.regularizer)

# each component adds its gradient on `rbm` (a plain or a standardized RBM)
function _∂regularize_composite!(∂::∂RBM, rbm, reg::CompositeRegularizer)
    foreach(r -> ∂regularize!(∂, rbm, r), reg.regularizers)
    return ∂
end

"""
    regularization_penalty(rbm, regularizer)

The penalty of `regularizer` (an [`AbstractRegularizer`](@ref)) on the parameters of `rbm`:
for a `StandardizedRBM`, on those of its equivalent plain `RBM` ([`unstandardize`](@ref)),
unless the regularizer is a [`StandardizedParametersRegularizer`](@ref).
"""
regularization_penalty(rbm::RBM, reg::L2FieldsRegularizer) = reg.λ / 2 * regularization_penalty_fields(rbm.visible)
regularization_penalty(rbm::RBM, reg::L1WeightsRegularizer) = reg.λ * sum(abs, rbm.w)
regularization_penalty(rbm::RBM, reg::L2WeightsRegularizer) = reg.λ / 2 * sum(abs2, rbm.w)

function regularization_penalty(rbm::RBM, reg::L2L1WeightsRegularizer)
    dims = ntuple(identity, ndims(rbm.visible))
    return reg.λ / (2 * length(rbm.visible)) * sum(abs2, sum(abs, rbm.w; dims))
end

regularization_penalty(rbm::RBM, reg::CompositeRegularizer) = _composite_penalty(rbm, reg)
regularization_penalty(rbm::RBM, reg::StandardizedParametersRegularizer) = regularization_penalty(rbm, reg.regularizer)

# the sum of the penalties of the components on `rbm` (a plain or a standardized RBM)
function _composite_penalty(rbm, reg::CompositeRegularizer)
    penalties = map(r -> regularization_penalty(rbm, r), reg.regularizers)
    return isempty(penalties) ? zero(eltype(rbm.w)) : sum(penalties)
end

# L2 penalty on the fields of a layer (`θ`, or `θp` and `θn` for dReLU), and its gradient
regularization_penalty_fields(layer::_ThetaLayers) = sum(abs2, layer.θ)
regularization_penalty_fields(layer::dReLU) = sum(abs2, layer.θp) + sum(abs2, layer.θn)

function ∂regularize_fields!(∂::AbstractArray, layer::_ThetaLayers, λ::Real)
    ∂[1, ..] .+= λ .* layer.θ
    return ∂
end

function ∂regularize_fields!(∂::AbstractArray, layer::dReLU, λ::Real)
    ∂[1, ..] .+= λ .* layer.θp
    ∂[2, ..] .+= λ .* layer.θn
    return ∂
end

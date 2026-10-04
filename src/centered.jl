"""
    CenteredRBM{V,H,W,Ov,Oh}

A [`StandardizedRBM`](@ref) whose scales are fixed to one, so in-place updates change only
its offsets. See <http://jmlr.org/papers/v17/14-237.html>.
"""
const CenteredRBM{V, H, W, Ov, Oh} = StandardizedRBM{V, H, W, Ov, Oh, <:Trues, <:Trues}

"""
    CenteredRBM(rbm, λv, λh)

Creates a centered RBM, with offsets `λv` (visible) and `λh` (hidden).
See <http://jmlr.org/papers/v17/14-237.html> for details.
The resulting model is *not* equivalent to the original `rbm`, unless `λv = 0` and `λh = 0`.
To construct an equivalent model instead, use [`standardize`](@ref).
"""
function CenteredRBM(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
    return StandardizedRBM(rbm, offset_v, offset_h, Trues(size(rbm.visible)), Trues(size(rbm.hidden)))
end

function CenteredRBM(
        visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray
    )
    return CenteredRBM(RBM(visible, hidden, w), offset_v, offset_h)
end

"""
    CenteredRBM(rbm)
    CenteredRBM(visible, hidden, w)

Creates a centered RBM, with offsets initialized to zero.
"""
CenteredRBM(rbm::RBM) = CenteredRBM(rbm, zeros_like(rbm.w, size(rbm.visible)), zeros_like(rbm.w, size(rbm.hidden)))
CenteredRBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray) = CenteredRBM(RBM(visible, hidden, w))
"""
    CenteredBinaryRBM(a, b, w, λv = 0, λh = 0)

Construct a centered binary RBM. The energy function is given by:

```math
E(v,h) = -a' * v - b' * h - (v - λv)' * w * (h - λh)
```
"""
function CenteredBinaryRBM(
        a::AbstractArray, b::AbstractArray, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray
    )
    return CenteredRBM(BinaryRBM(a, b, w), offset_v, offset_h)
end

function CenteredBinaryRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    return CenteredRBM(BinaryRBM(a, b, w))
end

# The scales of a `CenteredRBM` are fixed to one, so fitting statistics from data only
# updates its offsets.
function standardize_visible_from_data!(
        rbm::CenteredRBM, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), ϵ::Real = 0
    )
    return standardize_visible!(rbm, batchmean(rbm.visible, data; wts), rbm.scale_v)
end

function standardize_hidden_from_inputs!(
        rbm::CenteredRBM, inputs::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, inputs), damping::Real = 1, ϵ::Real = 0
    )
    μ = total_mean_from_inputs(rbm.hidden, inputs; wts)
    offset_h = (1 - damping) .* rbm.offset_h .+ damping .* μ
    return standardize_hidden!(rbm, offset_h, rbm.scale_h)
end

"""
    rescale_hidden!(rbm::CenteredRBM, λ::AbstractArray)

Scales parameters such that hidden unit activations are divided by `λ`, preserving
the modeled distribution. This assumes the hidden units have a scale parameter,
otherwise it does nothing and returns `false`. Since the interaction involves
`h - offset_h`, the hidden offsets are divided by `λ` together with the activations, and
since the unit scales stay fixed, the weights absorb the factor `λ`.
"""
function rescale_hidden!(rbm::CenteredRBM, λ::AbstractArray)
    @assert size(rbm.hidden) == size(λ)
    if rescale_activations!(rbm.hidden, λ)
        rbm.w .*= _along_hidden(rbm, λ)
        rbm.offset_h ./= λ
        return true
    end
    return false
end

# the unit scales cannot absorb the hidden scale gauge, so normalize the weights instead
rescale_hidden_activations!(rbm::CenteredRBM) = rescale_weights!(rbm)

"""
    _PlainStandardizedRBM{V,H,W}

A [`CenteredRBM`](@ref) whose offsets are also fixed to zero (lazy `Falses`), so it is
equivalent to the plain `RBM` with the same layers and weights. It is how [`pcd!`](@ref)
trains a plain `RBM`: fitting statistics from data leaves it unchanged, and the operations
that would involve the offsets or scales are those of the plain `RBM`.
"""
const _PlainStandardizedRBM{V, H, W} = StandardizedRBM{V, H, W, <:Falses, <:Falses, <:Trues, <:Trues}

# shares the layers and weights of `rbm`
function _PlainStandardizedRBM(rbm::RBM)
    offset_v, offset_h = Falses(size(rbm.visible)), Falses(size(rbm.hidden))
    scale_v, scale_h = Trues(size(rbm.visible)), Trues(size(rbm.hidden))
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

# The offsets and scales are fixed, so fitting statistics from data changes nothing.
function standardize_visible_from_data!(
        rbm::_PlainStandardizedRBM, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), ϵ::Real = 0
    )
    return rbm
end

function standardize_hidden_from_inputs!(
        rbm::_PlainStandardizedRBM, inputs::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, inputs), damping::Real = 1, ϵ::Real = 0
    )
    return rbm
end

function standardize_hidden_from_v!(
        rbm::_PlainStandardizedRBM, v::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, v), damping::Real = 1, ϵ::Real = 0
    )
    return rbm # skips computing the hidden inputs
end

# With zero offsets and unit scales the model is the plain `RBM` of its layers and weights,
# so the operations that would involve the offsets or scales reduce to those of the `RBM`.
unstandardize(rbm::_PlainStandardizedRBM) = RBM(rbm)
free_energy(rbm::_PlainStandardizedRBM, v::AbstractArray) = free_energy(RBM(rbm), v)
free_energy_h(rbm::_PlainStandardizedRBM, h::AbstractArray) = free_energy_h(RBM(rbm), h)
zerosum!(rbm::_PlainStandardizedRBM) = (zerosum!(RBM(rbm)); rbm)
zerosum!(∂::∂RBM, rbm::_PlainStandardizedRBM) = zerosum!(∂, RBM(rbm))
rescale_hidden!(rbm::_PlainStandardizedRBM, λ::AbstractArray) = rescale_hidden!(RBM(rbm), λ)

function ∂regularize!(∂::∂RBM, rbm::_PlainStandardizedRBM; regularize_unstandardized::Bool = true, kwargs...)
    return ∂regularize!(∂, RBM(rbm); kwargs...) # standardized and unstandardized parameters coincide
end

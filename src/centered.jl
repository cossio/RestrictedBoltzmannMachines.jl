"""
    CenteredRBM{V,H,W,Ov,Oh}

A [`StandardizedRBM`](@ref) whose scales are lazy ones (`FillArrays.Ones`, usually `Trues`),
so they are fixed and in-place updates change only its offsets.
See <http://jmlr.org/papers/v17/14-237.html>.
"""
const CenteredRBM{V, H, W, Ov, Oh} = StandardizedRBM{V, H, W, Ov, Oh, <:Ones, <:Ones}

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
        _maybe_div!(rbm.offset_h, λ)
        return true
    end
    return false
end

# the unit scales cannot absorb the hidden scale gauge, so normalize the weights instead
rescale_hidden_activations!(rbm::CenteredRBM) = rescale_weights!(rbm)

"""
    PlainStandardizedRBM{V,H,W}

A [`CenteredRBM`](@ref) whose offsets are also lazy zeros (`FillArrays.Zeros`, usually
`Falses`), so it is equivalent to the plain `RBM` with the same layers and weights.
[`pcd!`](@ref) trains a plain `RBM` through it.
"""
const PlainStandardizedRBM{V, H, W} = StandardizedRBM{V, H, W, <:Zeros, <:Zeros, <:Ones, <:Ones}

# shares the layers and weights of `rbm`
PlainStandardizedRBM(rbm::RBM) = StandardizedRBM(
    rbm, Falses(size(rbm.visible)), Falses(size(rbm.hidden)), Trues(size(rbm.visible)), Trues(size(rbm.hidden))
)

# The offsets and scales are fixed, so fitting statistics from data changes nothing.
standardize_visible_from_data!(rbm::PlainStandardizedRBM, data::AbstractArray; kwargs...) = rbm
standardize_hidden_from_inputs!(rbm::PlainStandardizedRBM, inputs::AbstractArray; kwargs...) = rbm
standardize_hidden_from_v!(rbm::PlainStandardizedRBM, v::AbstractArray; kwargs...) = rbm # skips the inputs

unstandardize(rbm::PlainStandardizedRBM) = RBM(rbm)
free_energy(rbm::PlainStandardizedRBM, v::AbstractArray) = free_energy(RBM(rbm), v)
free_energy_h(rbm::PlainStandardizedRBM, h::AbstractArray) = free_energy_h(RBM(rbm), h)

# standardized and unstandardized parameters coincide
_∂regularize_unstandardized!(∂::∂RBM, rbm::PlainStandardizedRBM, reg::AbstractRegularizer) =
    ∂regularize!(∂, RBM(rbm), reg)

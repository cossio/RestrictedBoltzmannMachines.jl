struct CenteredRBM{V,H,W,Ov,Oh}
    visible::V
    hidden::H
    w::W
    offset_v::Ov
    offset_h::Oh
    function CenteredRBM{V,H,W,Ov,Oh}(
        visible::V, hidden::H, w::W, λv::Ov, λh::Oh
    ) where {V<:AbstractLayer, H<:AbstractLayer, W<:AbstractArray, Ov<:AbstractArray, Oh<:AbstractArray}
        @assert size(w) == (size(visible)..., size(hidden)...)
        @assert size(visible) == size(λv)
        @assert size(hidden) == size(λh)
        return new{V,H,W,Ov,Oh}(visible, hidden, w, λv, λh)
    end
end

function CenteredRBM(
    visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray
)
    V, H, W = typeof(visible), typeof(hidden), typeof(w)
    Ov, Oh = typeof(offset_v), typeof(offset_h)
    return CenteredRBM{V,H,W,Ov,Oh}(visible, hidden, w, offset_v, offset_h)
end

"""
    CenteredRBM(rbm, λv, λh)

Creates a centered RBM, with offsets `λv` (visible) and `λh` (hidden).
See <http://jmlr.org/papers/v17/14-237.html> for details.
The resulting model is *not* equivalent to the original `rbm`, unless `λv = 0` and `λh = 0`.
"""
function CenteredRBM(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
    return CenteredRBM(rbm.visible, rbm.hidden, rbm.w, offset_v, offset_h)
end

"""
    CenteredRBM(visible, hidden, w)

Creates a centered RBM, with offsets initialized to zero.
"""
function CenteredRBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray)
    offset_v = similar(w, size(visible)) .= 0
    offset_h = similar(w, size(hidden)).= 0
    return CenteredRBM(RBM(visible, hidden, w), offset_v, offset_h)
end

CenteredRBM(rbm::RBM) = CenteredRBM(rbm.visible, rbm.hidden, rbm.w)
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

"""
    RBM(centered_rbm::CenteredRBM)

Returns an (uncentered) `RBM` which neglects the offsets of `centered_rbm`.
The resulting model is *not* equivalent to the original `centered_rbm`.
To construct an equivalent model, use the function
`uncenter(centered_rbm)` instead (see [`uncenter`](@ref)).
Shares parameters with `centered_rbm`.
"""
function RBM(rbm::CenteredRBM)
    return RBM(rbm.visible, rbm.hidden, rbm.w)
end

function energy(rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(rbm.visible, v)
    Eh = energy(rbm.hidden, h)
    Ew = interaction_energy(rbm, v, h)
    return Ev .+ Eh .+ Ew
end

function interaction_energy(centered_rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    centered_v = v .- centered_rbm.offset_v
    centered_h = h .- centered_rbm.offset_h
    return interaction_energy(RBM(centered_rbm), centered_v, centered_h)
end

function inputs_h_from_v(centered_rbm::CenteredRBM, v::AbstractArray)
    centered_v = v .- centered_rbm.offset_v
    return inputs_h_from_v(RBM(centered_rbm), centered_v)
end

function inputs_v_from_h(centered_rbm::CenteredRBM, h::AbstractArray)
    centered_h = h .- centered_rbm.offset_h
    return inputs_v_from_h(RBM(centered_rbm), centered_h)
end

function free_energy(rbm::CenteredRBM, v::AbstractArray)
    E = energy(rbm.visible, v)
    inputs = inputs_h_from_v(rbm, v)
    F = -cgf(rbm.hidden, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_h), inputs)
    return E + F - ΔE
end

function mean_h_from_v(rbm::CenteredRBM, v::AbstractArray)
    inputs = inputs_h_from_v(rbm, v)
    return mean_from_inputs(rbm.hidden, inputs)
end

function mean_v_from_h(rbm::CenteredRBM, h::AbstractArray)
    inputs = inputs_v_from_h(rbm, h)
    return mean_from_inputs(rbm.visible, inputs)
end

function ∂free_energy(
    rbm::CenteredRBM, v::AbstractArray; wts = nothing,
    moments = moments_from_samples(rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(rbm, v)
    ∂v = ∂energy_from_moments(rbm.visible, moments)

    ∂Γ = ∂cgfs(rbm.hidden, inputs)
    h = grad2ave(rbm.hidden, ∂Γ)

    dims = (ndims(rbm.hidden.par) + 1):ndims(∂Γ)
    ∂h = reshape(wmean(-∂Γ; wts, dims), size(rbm.hidden.par))

    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return ∂RBM(∂v, ∂h, ∂w)
end

function ∂interaction_energy(
    rbm::CenteredRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    centered_v = v .- rbm.offset_v
    centered_h = h .- rbm.offset_h
    ∂w = ∂interaction_energy(RBM(rbm), centered_v, centered_h; wts)
    return ∂w
end

function log_pseudolikelihood(rbm::CenteredRBM, v::AbstractArray)
    return log_pseudolikelihood(uncenter(rbm), v)
end

function mirror(rbm::CenteredRBM)
    perm = ntuple(Val(ndims(rbm.w))) do i
        if i ≤ ndims(rbm.hidden)
            i + ndims(rbm.visible)
        else
            i - ndims(rbm.hidden)
        end
    end
    w = permutedims(rbm.w, perm)
    return CenteredRBM(rbm.hidden, rbm.visible, w, rbm.offset_h, rbm.offset_v)
end


@doc raw"""
    uncenter(centered_rbm::CenteredRBM)

Constructs an `RBM` equivalent to the given `CenteredRBM`.
The energies assigned by the two models differ by a constant amount,

```math
E(v,h) - E_c(v,h) = \sum_{i\mu}w_{i\mu}\lambda_i\lambda_\mu
```

where ``E_c(v,h)`` is the energy assigned by `centered_rbm` and ``E(v,h)`` is the energy
assigned by the `RBM` constructed by this method.

This is the inverse operation of [`center`](@ref).

To construct an `RBM` that simply neglects the offsets, call `RBM(centered_rbm)` instead.
"""
uncenter(centered_rbm::CenteredRBM) = RBM(center(centered_rbm))
uncenter(rbm::RBM) = rbm

@doc raw"""
    center(rbm::RBM, offset_v = 0, offset_h = 0)

Constructs a `CenteredRBM` equivalent to the given `rbm`.
The energies assigned by the two models differ by a constant amount,

```math
E(v,h) - E_c(v,h) = \sum_{i\mu}w_{i\mu}\lambda_i\lambda_\mu
```

where ``E(v,h)`` is the energy assigned by the original `rbm`, and
``E_c(v,h)`` is the energy assigned by the returned `CenteredRBM`.

This is the inverse operation of [`uncenter`](@ref).

To construct a `CenteredRBM` that simply includes these offsets,
call `CenteredRBM(rbm, offset_v, offset_h)` instead.
"""
function center(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
    centered_rbm = center(rbm)
    return center(centered_rbm, offset_v, offset_h)
end

function center(rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray)
    center!(deepcopy(rbm), offset_v, offset_h)
end

center(rbm::CenteredRBM) = center!(deepcopy(rbm))
center(rbm::RBM) = CenteredRBM(rbm)

"""
    center!(centered_rbm, offset_v = 0, offset_h = 0)

Transforms the offsets of `centered_rbm`. The transformed model is equivalent to
the original one (energies differ by a constant).
"""
function center!(rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray)
    center_visible!(rbm, offset_v)
    center_hidden!(rbm, offset_h)
    return rbm
end

function center!(rbm::CenteredRBM)
    offset_v = Zeros{eltype(rbm.w)}(size(rbm.visible))
    offset_h = Zeros{eltype(rbm.w)}(size(rbm.hidden))
    return center!(rbm, offset_v, offset_h)
end

function center_visible!(rbm::CenteredRBM, offset_v::AbstractArray)
    inputs = inputs_h_from_v(rbm, offset_v)
    shift_fields!(rbm.hidden, inputs)
    rbm.offset_v .= offset_v
    return rbm
end

function center_hidden!(rbm::CenteredRBM, offset_h::AbstractArray)
    inputs = inputs_v_from_h(rbm, offset_h)
    shift_fields!(rbm.visible, inputs)
    rbm.offset_h .= offset_h
    return rbm
end

function center_visible_from_data!(rbm::CenteredRBM, data::AbstractArray; wts=nothing)
    offset_v = batchmean(rbm.visible, data; wts)
    return center_visible!(rbm, offset_v)
end

function center_hidden_from_data!(rbm::CenteredRBM, data::AbstractArray; wts=nothing)
    h = mean_h_from_v(rbm, data)
    offset_h = batchmean(rbm.hidden, h; wts)
    return center_hidden!(rbm, offset_h)
end

function center_from_data!(rbm::CenteredRBM, data::AbstractArray; wts=nothing)
    center_visible_from_data!(rbm, data; wts)
    center_hidden_from_data!(rbm, data; wts)
    return rbm
end

function zerosum!(rbm::CenteredRBM)
    zerosum!(RBM(rbm))
    return rbm
end

function rescale_weights!(rbm::CenteredRBM)
    rescale_weights!(RBM(rbm))
    return rbm
end

function initialize!(rbm::CenteredRBM, data::AbstractArray; ϵ::Real = 1e-6)
    initialize!(RBM(rbm), data; ϵ)
    center_from_data!(rbm, data)
    return rbm
end

function ∂regularize!(∂::∂RBM, rbm::CenteredRBM; kwargs...)
    ∂regularize!(∂, RBM(rbm); kwargs...)
end

function sample_h_from_v(rbm::CenteredRBM, v::AbstractArray)
    inputs = inputs_h_from_v(rbm, v)
    return sample_from_inputs(rbm.hidden, inputs)
end

function sample_v_from_h(rbm::CenteredRBM, h::AbstractArray)
    inputs = inputs_v_from_h(rbm, h)
    return sample_from_inputs(rbm.visible, inputs)
end

function sample_v_from_v(rbm::CenteredRBM, v::AbstractArray; steps::Int = 1)
    for _ in 1:steps
        v = oftype(v, sample_v_from_v_once(rbm, v))
    end
    return v
end

function sample_h_from_h(rbm::CenteredRBM, h::AbstractArray; steps::Int = 1)
    for _ in 1:steps
        h = oftype(h, sample_h_from_h_once(rbm, h))
    end
    return h
end

function sample_v_from_v_once(rbm::CenteredRBM, v::AbstractArray)
    h = sample_h_from_v(rbm, v)
    v = sample_v_from_h(rbm, h)
    return v
end

function sample_h_from_h_once(rbm::CenteredRBM, h::AbstractArray)
    v = sample_v_from_h(rbm, h)
    h = sample_h_from_v(rbm, v)
    return h
end

function pcd!(
    rbm::CenteredRBM,
    data::AbstractArray;

    batchsize::Int = 1,
    iters::Int = 1,

    optim::AbstractRule = Adam(), # a rule from Optimisers
    steps::Int = 1, # Monte-Carlo steps to update persistent chains

    # data point weights
    wts::Union{AbstractVector, Nothing} = nothing,

    # init fantasy chains
    vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),

    moments = moments_from_samples(rbm.visible, data; wts),

    # damping to update hidden statistics
    hidden_offset_damping::Real = 1//100,

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    rescale::Bool = true, # normalize weights to unit norm (for continuous hidden units only)

    callback = Returns(nothing)
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    isnothing(wts) || @assert size(data)[end] == length(wts)

    # inital centering from data
    center_from_data!(rbm, data)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    # define parameters for Optimiser
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)

    wts_mean = isnothing(wts) ? 1 : mean(wts)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        # centering
        offset_h_new = grad2ave(rbm.hidden, -∂d.hidden) # <h>_d from minibatch
        offset_h = (1 - hidden_offset_damping) * rbm.offset_h + hidden_offset_damping * offset_h_new
        center_hidden!(rbm, offset_h)

        # gauge constraints
        zerosum && zerosum!(rbm)
        rescale && rescale_weights!(rbm)

        callback(; rbm, optim, iter, vm, vd, wd)
    end
    return state, ps
end

"""
    RBM

Represents a restricted Boltzmann Machine.
"""
struct RBM{V, H, W<:AbstractArray}
    visible::V
    hidden::H
    weights::W
    function RBM(visible::V, hidden::H, weights::W) where {V, H, W<:AbstractArray}
        @assert size(weights) == (size(visible)..., size(hidden)...)
        return new{V,H,W}(visible, hidden, weights)
    end
end

Flux.@functor RBM

"""
    energy(rbm, v, h)

Energy of the rbm in the configuration `(v,h)`.
"""
function energy(rbm::RBM, v::AbstractArray, h::AbstractArray)
    scalar_case = (
        size(v) == size(rbm.visible) && size(h) == size(rbm.hidden)
    )
    batches_case = (
        size(v) == (size(rbm.visible)..., size(v)[end]) &&
        size(h) == (size(rbm.hidden)..., size(h)[end])
    )
    @assert scalar_case || batches_case
    Ev = energy(rbm.visible, v)
    Eh = energy(rbm.hidden, h)
    Ew = interaction_energy(rbm, v, h)
    return Ev + Eh + Ew
end

"""
    interaction_energy(rbm, v, h)

Weight mediated interaction energy.
"""
function interaction_energy(rbm::RBM, v::AbstractArray, h::AbstractArray)
    wmat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    vflat = flatten(rbm.visible, v)
    hflat = flatten(rbm.hidden, h)
    return flat_interaction_energy(wmat, vflat, hflat)
end

function flat_interaction_energy(w::AbstractMatrix, v::AbstractVector, h::AbstractVector)
    @assert size(w) == (length(v), length(h))
    return dot(v, w, h)
end

function flat_interaction_energy(w::AbstractMatrix, v::AbstractMatrix, h::AbstractMatrix)
    @assert size(w) == (size(v, 1), size(h, 1))
    @assert size(v, 2) == size(h, 2)
    if size(v, 1) ≥ size(h, 1)
        E = -sum_((w' * v) .* h; dims=1)
    else
        E = -sum_(v .* (w * h); dims=1)
    end
    @assert length(E::AbstractVector) == size(v, 2) == size(h, 2)
    return E
end

"""
    inputs_v_to_h(rbm, v)

Interaction inputs from visible to hidden layer.
"""
function inputs_v_to_h(rbm::RBM, v::AbstractArray)
    vflat = flatten(rbm.visible, v)
    wflat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    w_, v_ = matmul_convert_maybe(wflat, vflat)
    return unflatten(rbm.hidden, w_' * v_)
end

"""
    inputs_h_to_v(rbm, h)

Interaction inputs from hidden to visible layer.
"""
function inputs_h_to_v(rbm::RBM, h::AbstractArray)
    hflat = flatten(rbm.hidden, h)
    wflat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    w_, h_ = matmul_convert_maybe(wflat, hflat)
    return unflatten(rbm.visible, w_ * h_)
end

function flatten(layer, x::AbstractArray)
    case_example = size(x) == size(layer)
    case_batches = size(x) == (size(layer)..., size(x)[end])
    @assert case_example || case_batches
    if case_example
        return reshape(x, length(layer))
    else
        return reshape(x, length(layer), size(x)[end])
    end
end

unflatten(layer, x::AbstractVector) = reshape(x, size(layer))
unflatten(layer, x::AbstractMatrix) = reshape(x, size(layer)..., size(x, 2))

# convert to common eltype before matrix multiply, to make sure we hit BLAS
function matmul_convert_maybe(x::AbstractArray{X}, y::AbstractArray{Y}) where {X,Y}
    T = promote_type(X, Y)
    return T.(x), T.(y)
end
matmul_convert_maybe(x::AbstractArray{T}, y::AbstractArray{T}) where {T} = x, y

"""
    free_energy(rbm, v; β=1)

Free energy of visible configuration (after marginalizing hidden configurations).
"""
function free_energy(rbm::RBM, v::AbstractArray; β::Real = 1)
    E = energy(rbm.visible, v)
    inputs = inputs_v_to_h(rbm, v)
    F = free_energy(rbm.hidden, inputs; β)
    return E + F
end

"""
    sample_h_from_v(rbm, v; β=1)

Samples a hidden configuration conditional on the visible configuration `v`.
"""
function sample_h_from_v(rbm::RBM, v::AbstractArray; β::Real = 1)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_sample(rbm.hidden, inputs; β)
end

"""
    sample_v_from_h(rbm, h; β = 1)

Samples a visible configuration conditional on the hidden configuration `h`.
"""
function sample_v_from_h(rbm::RBM, h::AbstractArray; β::Real = 1)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_sample(rbm.visible, inputs; β)
end

"""
    sample_v_from_v(rbm, v; β = 1, steps = 1)

Samples a visible configuration conditional on another visible configuration `v`.
"""
function sample_v_from_v(rbm::RBM, v::AbstractArray; β::Real = 1, steps::Int = 1)
    @assert size(v) == size(rbm.visible) || size(v) == (size(rbm.visible)..., size(v)[end])
    @assert steps ≥ 1
    v_ = sample_v_from_v_once(rbm, v; β)
    for _ in 1:(steps - 1)
        v_ = sample_v_from_v_once(rbm, v_; β)
    end
    return v_
end

"""
    sample_h_from_h(rbm, h; β = 1, steps = 1)

Samples a hidden configuration conditional on another hidden configuration `h`.
"""
function sample_h_from_h(rbm::RBM, h::AbstractArray; β::Real = 1, steps::Int = 1)
    @assert size(h) == size(rbm.hidden) || size(h) == (size(rbm.hidden)..., size(h)[end])
    @assert steps ≥ 1
    h_ = sample_h_from_h_once(rbm, h; β)
    for _ in 1:(steps - 1)
        h_ = sample_h_from_h_once(rbm, h_; β)
    end
    return h_
end

function sample_v_from_v_once(rbm::RBM, v::AbstractArray; β::Real = 1)
    h = sample_h_from_v(rbm, v; β)
    v = sample_v_from_h(rbm, h; β)
    return v
end

function sample_h_from_h_once(rbm::RBM, h::AbstractArray; β::Real = 1)
    v = sample_v_from_h(rbm, h; β)
    h = sample_h_from_v(rbm, v; β)
    return h
end

"""
    mean_h_from_v(rbm, v; β = 1)

Mean unit activation values, conditioned on the other layer, <h | v>.
"""
function mean_h_from_v(rbm::RBM, v::AbstractArray; β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_mean(rbm.hidden, inputs; β)
end

"""
    mean_v_from_h(rbm, v; β = 1)

Mean unit activation values, conditioned on the other layer, <v | h>.
"""
function mean_v_from_h(rbm::RBM, h::AbstractArray; β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_mean(rbm.visible, inputs; β)
end

"""
    mode_v_from_h(rbm, h)

Mode unit activations, conditioned on the other layer.
"""
function mode_v_from_h(rbm::RBM, h::AbstractArray)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_mode(rbm.visible, inputs)
end

"""
    mode_h_from_v(rbm, v)

Mode unit activations, conditioned on the other layer.
"""
function mode_h_from_v(rbm::RBM, v::AbstractArray)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_mode(rbm.hidden, inputs)
end

function ∂weights(rbm::RBM, v::AbstractArray, h::AbstractArray = mean_h_from_v(rbm, v))
    @assert size(v) == size(rbm.visible) || size(v) == (size(rbm.visible)..., size(v)[end])
    @assert size(h) == size(rbm.hidden)  || size(h) == (size(rbm.hidden)...,  size(h)[end])
    @assert ndims(v) - ndims(rbm.visible) == ndims(h) - ndims(rbm.hidden)
    @assert size(v, ndims(rbm.visible) + 1) == size(h, ndims(rbm.hidden) + 1)
    v_flat = flatten(rbm.visible, v)
    h_flat = flatten(rbm.hidden, h)
    ∂w = v_flat * h_flat' / size(v_flat, 2)
    return reshape(∂w, size(rbm.weights))
end

function ∂free_energy(
    rbm::RBM, v::AbstractArray, inputs::AbstractArray = inputs_v_to_h(rbm, v)
)
    h = transfer_mean(rbm.hidden, inputs)
    ∂w = ∂weights(rbm, v, h)
    ∂v = ∂energy(rbm.visible, v)
    ∂h = ∂free_energy(rbm.hidden, inputs)
    return (visible = ∂v, hidden = ∂h, weights = ∂w)
end

"""
    reconstruction_error(rbm, v; β = 1, steps = 1)

Stochastic reconstruction error of `v`.
"""
function reconstruction_error(rbm::RBM, v::AbstractArray; β::Real = 1, steps::Int = 1)
    v_ = sample_v_from_v(rbm, v; β, steps)
    return mean_(abs.(v .- v_); dims = layerdims(rbm.visible))
end

"""
    flip_layers(rbm)

Returns a new RBM with viible and hidden layers flipped.
"""
function flip_layers(rbm::RBM)
    function p(i)
        if i ≤ ndims(rbm.visible)
            return i + ndims(rbm.hidden)
        else
            return i - ndims(rbm.visible)
        end
    end
    perm = ntuple(p, ndims(rbm.weights))
    return RBM(rbm.hidden, rbm.visible, permutedims(rbm.weights, perm))
end

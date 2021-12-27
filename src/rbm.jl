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
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    @assert size(h) == (size(rbm.hidden)...,  size(h)[end])
    wmat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    vmat = reshape(v, length(rbm.visible), :)
    hmat = reshape(h, length(rbm.hidden), :)
    if length(rbm.visible) ≥ length(rbm.hidden)
        E = -sum((wmat' * vmat) .* hmat; dims=1)
    else
        E = -sum(vmat .* (wmat * hmat); dims=1)
    end
    return reshape(E, last(size(vmat)))
end

"""
    inputs_v_to_h(rbm, v)

Interaction inputs from visible to hidden layer.
"""
function inputs_v_to_h(rbm::RBM, v::AbstractArray)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    wmat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    vmat = reshape(v, length(rbm.visible), :)
    # convert to common eltype to make sure we hit BLAS
    return reshape(wmat' * oftype(wmat, vmat), size(rbm.hidden)..., :)
end

"""
    inputs_h_to_v(rbm, h)

Interaction inputs from hidden to visible layer.
"""
function inputs_h_to_v(rbm::RBM, h::AbstractArray)
    @assert size(h) == (size(rbm.hidden)..., size(h)[end])
    wmat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    hmat = reshape(h, length(rbm.hidden), :)
    # convert to common eltype to make sure we hit BLAS
    return reshape(wmat * oftype(wmat, hmat), size(rbm.visible)..., :)
end

"""
    free_energy(rbm, v, β=1)

Free energy of visible configuration (after marginalizing hidden configurations).
"""
function free_energy(rbm::RBM, v::AbstractArray, β::Real = true)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    E = energy(rbm.visible, v)
    inputs = inputs_v_to_h(rbm, v)
    Γ = cgf(rbm.hidden, inputs, β)
    return E - Γ
end

"""
    sample_h_from_v(rbm, v, β=1)

Samples a hidden configuration conditional on the visible configuration `v`.
"""
function sample_h_from_v(rbm::RBM, v::AbstractArray, β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_sample(rbm.hidden, inputs, β)
end

"""
    sample_v_from_h(rbm, h, β = 1)

Samples a visible configuration conditional on the hidden configuration `h`.
"""
function sample_v_from_h(rbm::RBM, h::AbstractArray, β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_sample(rbm.visible, inputs, β)
end

"""
    sample_v_from_v(rbm, v, β = 1; steps = 1)

Samples a visible configuration conditional on another visible configuration `v`.
"""
function sample_v_from_v(rbm::RBM, v::AbstractArray, β::Real = true; steps::Int = 1)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    @assert steps ≥ 1
    v_ = sample_v_from_v_once(rbm, v, β)
    for _ in 1:(steps - 1)
        v_ = sample_v_from_v_once(rbm, v_, β)
    end
    return v_
end

"""
    sample_h_from_h(rbm, h, β = 1; steps = 1)

Samples a hidden configuration conditional on another hidden configuration `h`.
"""
function sample_h_from_h(rbm::RBM, h::AbstractArray, β::Real = true; steps::Int = 1)
    @assert size(h) == (size(rbm.hidden)..., size(h)[end])
    @assert steps ≥ 1
    h_ = sample_h_from_h_once(rbm, h, β)
    for _ in 1:(steps - 1)
        h_ = sample_h_from_h_once(rbm, h_, β)
    end
    return h_
end

function sample_v_from_v_once(rbm::RBM, v::AbstractArray, β::Real = true)
    h = sample_h_from_v(rbm, v, β)
    v = sample_v_from_h(rbm, h, β)
    return v
end

function sample_h_from_h_once(rbm::RBM, h::AbstractArray, β::Real = true)
    v = sample_v_from_h(rbm, h, β)
    h = sample_h_from_v(rbm, v, β)
    return h
end

"""
    mean_h_from_v(rbm, v, β = 1)

Mean unit activation values, conditioned on the other layer, <h | v>.
"""
function mean_h_from_v(rbm::RBM, v::AbstractArray, β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_mean(rbm.hidden, inputs, β)
end

"""
    conjugates_v_from_h(rbm, v, β = 1)

Mean unit activation values, conditioned on the other layer, <v | h>.
"""
function mean_v_from_h(rbm::RBM, h::AbstractArray, β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_mean(rbm.visible, inputs, β)
end

function conjugates_h_from_v(rbm::RBM, v::AbstractArray, β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return conjugates(rbm.hidden, inputs, β)
end

function conjugates_v_from_h(rbm::RBM, h::AbstractArray, β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return conjugates(rbm.visible, inputs, β)
end

"""
    reconstruction_error(rbm, v, β = 1; steps = 1)

Stochastic reconstruction error of `v`.
"""
function reconstruction_error(rbm::RBM, v::AbstractArray, β::Real = true; steps::Int = 1)
    v_ = sample_v_from_v(rbm, v, β; steps = steps)
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

function convergence_score(rbm::RBM, data::AbstractArray; steps::Int)
    v = sample_v_from_v(rbm, data; steps=steps)
    h = sample_h_from_v(rbm, v)
    data_h = sample_h_from_v(rbm, data)

    conjugates_v_from_h(rbm, h)
    conjugates_v_from_h(rbm, data_h)

    conjugates_h_from_v(rbm, v)
    conjugates_h_from_v(rbm, data)

    ∂w_model = reshape(v, length(rbm.visible), :) * reshape(h, length(rbm.hidden), :)' / size(data)[end]
    ∂w_data = reshape(data, length(rbm.visible), :) * reshape(data_h, length(rbm.hidden), :)' / size(data)[end]


end

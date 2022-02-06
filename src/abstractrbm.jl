"""
    AbstractRBM{V,H,W}

Abstract RBM type, with visible layer of type `V`, hidden layer of type `H`, and weights
of type `W`.
"""
abstract type AbstractRBM{V,H,W} end

flat_w(rbm::AbstractRBM) = reshape(weights(rbm), length(visible(rbm)), length(hidden(rbm)))
flat_v(rbm::AbstractRBM, v::AbstractArray) = flatten(visible(rbm), v)
flat_h(rbm::AbstractRBM, h::AbstractArray) = flatten(hidden(rbm), h)

function batchsize(rbm::AbstractRBM, v::AbstractArray, h::AbstractArray)
    visible_batchsize = batchsize(visible(rbm), v)
    hidden_batchsize = batchsize(hidden(rbm), h)
    if size(hidden(rbm)) == size(h)
        return visible_batchsize
    elseif size(visible(rbm)) == size(v)
        return hidden_batchsize
    else
        @assert visible_batchsize == hidden_batchsize
        return visible_batchsize
    end
end

"""
    energy(rbm, v, h)

Energy of the rbm in the configuration `(v,h)`.
"""
function energy(rbm::AbstractRBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(visible(rbm), v)
    Eh = energy(hidden(rbm), h)
    Ew = interaction_energy(rbm, v, h)
    return Ev .+ Eh .+ Ew
end

"""
    free_energy(rbm, v; β = 1)

Free energy of visible configuration (after marginalizing hidden configurations).
"""
function free_energy(rbm::AbstractRBM, v::AbstractArray; β::Real = true)
    E = energy(visible(rbm), v)
    inputs = inputs_v_to_h(rbm, v)
    F = free_energy(hidden(rbm), inputs; β)
    return E + F
end

"""
    sample_h_from_v(rbm, v; β = 1)

Samples a hidden configuration conditional on the visible configuration `v`.
"""
function sample_h_from_v(rbm::AbstractRBM, v::AbstractArray; β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_sample(hidden(rbm), inputs; β)
end

"""
    sample_v_from_h(rbm, h; β = 1)

Samples a visible configuration conditional on the hidden configuration `h`.
"""
function sample_v_from_h(rbm::AbstractRBM, h::AbstractArray; β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_sample(visible(rbm), inputs; β)
end

"""
    sample_v_from_v(rbm, v; β = 1, steps = 1)

Samples a visible configuration conditional on another visible configuration `v`.
"""
function sample_v_from_v(rbm::AbstractRBM, v::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    v1 = copy(v)
    for _ in 1:steps
        v1 .= sample_v_from_v_once(rbm, v1; β)
    end
    return v1
end

"""
    sample_h_from_h(rbm, h; β = 1, steps = 1)

Samples a hidden configuration conditional on another hidden configuration `h`.
"""
function sample_h_from_h(rbm::AbstractRBM, h::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(rbm.hidden) == size(h)[1:ndims(hidden(rbm))]
    h1 = copy(h)
    for _ in 1:steps
        h1 .= sample_h_from_h_once(rbm, h1; β)
    end
    return h1
end

function sample_v_from_v_once(rbm::AbstractRBM, v::AbstractArray; β::Real = true)
    h = sample_h_from_v(rbm, v; β)
    v = sample_v_from_h(rbm, h; β)
    return v
end

function sample_h_from_h_once(rbm::AbstractRBM, h::AbstractArray; β::Real = true)
    v = sample_v_from_h(rbm, h; β)
    h = sample_h_from_v(rbm, v; β)
    return h
end

"""
    mean_h_from_v(rbm, v; β = 1)

Mean unit activation values, conditioned on the other layer, <h | v>.
"""
function mean_h_from_v(rbm::AbstractRBM, v::AbstractArray; β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_mean(hidden(rbm), inputs; β)
end

"""
    mean_v_from_h(rbm, v; β = 1)

Mean unit activation values, conditioned on the other layer, <v | h>.
"""
function mean_v_from_h(rbm::AbstractRBM, h::AbstractArray; β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_mean(visible(rbm), inputs; β)
end

"""
    mode_v_from_h(rbm, h)

Mode unit activations, conditioned on the other layer.
"""
function mode_v_from_h(rbm::AbstractRBM, h::AbstractArray)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_mode(visible(rbm), inputs)
end

"""
    mode_h_from_v(rbm, v)

Mode unit activations, conditioned on the other layer.
"""
function mode_h_from_v(rbm::AbstractRBM, v::AbstractArray)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_mode(hidden(rbm), inputs)
end

"""
    reconstruction_error(rbm, v; β = 1, steps = 1)

Stochastic reconstruction error of `v`.
"""
function reconstruction_error(rbm::AbstractRBM, v::AbstractArray; β::Real=true, steps::Int=1)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    v1 = sample_v_from_v(rbm, v; β, steps)
    ϵ = Statistics.mean(abs.(v .- v1); dims = 1:ndims(visible(rbm)))
    if ndims(v) == ndims(visible(rbm))
        return only(ϵ)
    else
        return reshape(ϵ, batchsize(visible(rbm), v))
    end
end

"""
    ∂free_energy(rbm, v)

Gradient of `free_energy(rbm, v)` with respect to model parameters.
If `v` consists of multiple samples (batches), then an average is taken.
"""
function ∂free_energy(
    rbm::AbstractRBM, v::AbstractArray;
    wts = nothing, stats = suffstats(visible(rbm), v; wts)
)
    inputs = inputs_v_to_h(rbm, v)
    h = transfer_mean(hidden(rbm), inputs)
    ∂v = ∂energy(visible(rbm), stats)
    ∂h = ∂free_energy(hidden(rbm), inputs; wts)
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

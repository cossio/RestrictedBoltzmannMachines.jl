"""
    AbstractRBM{V,H,W}

Abstract RBM type, with visible layer of type `V`, hidden layer of type `H`, and weights
of type `W`.
"""
abstract type AbstractRBM{V,H,W} end

flat_w(rbm::AbstractRBM) = reshape(weights(rbm), length(visible(rbm)), length(hidden(rbm)))
flat_v(rbm::AbstractRBM, v::AbstractArray) = flatten(visible(rbm), v)
flat_h(rbm::AbstractRBM, h::AbstractArray) = flatten(hidden(rbm), h)

"""
    batch_size(rbm, v, h)

Returns the batch size if `energy(rbm, v, h)` were computed.
"""
function batch_size(rbm::AbstractRBM, v::AbstractArray, h::AbstractArray)
    v_bsz = batch_size(visible(rbm), v)
    h_bsz = batch_size(hidden(rbm), h)
    if isempty(v_bsz)
        return h_bsz
    elseif isempty(h_bsz)
        return v_bsz
    else
        return join_batch_size(v_bsz, h_bsz)
    end
end

function join_batch_size(bsz_1::Tuple{Int,Vararg{Int}}, bsz_2::Tuple{Int,Vararg{Int}})
    if length(bsz_1) > length(bsz_2)
        D = length(bsz_2)
        sz2 = bsz_1[(D + 1):end]
    else
        D = length(bsz_1)
        sz2 = bsz_2[(D + 1):end]
    end
    sz1 = map(bsz_1[1:D], bsz_2[1:D]) do b1, b2
        bmin, bmax = minmax(b1, b2)
        @assert bmin == 1 || bmin == bmax
        bmax
    end
    return (sz1..., sz2...)
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
Ensures type stability by requiring that the returned array is of the same type as `v`.
"""
function sample_v_from_v(rbm::AbstractRBM, v::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    for _ in 1:steps
        v = oftype(v, sample_v_from_v_once(rbm, v; β))
    end
    return v
end

"""
    sample_h_from_h(rbm, h; β = 1, steps = 1)

Samples a hidden configuration conditional on another hidden configuration `h`.
Ensures type stability by requiring that the returned array is of the same type as `h`.
"""
function sample_h_from_h(rbm::AbstractRBM, h::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(hidden(rbm)) == size(h)[1:ndims(hidden(rbm))]
    for _ in 1:steps
        h = oftype(h, sample_h_from_h_once(rbm, h; β))
    end
    return h
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
    var_v_from_h(rbm, v; β = 1)

Variance of unit activation values, conditioned on the other layer, var(v | h).
"""
function var_v_from_h(rbm::AbstractRBM, h::AbstractArray; β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_var(visible(rbm), inputs; β)
end

"""
    var_h_from_v(rbm, v; β = 1)

Variance of unit activation values, conditioned on the other layer, var(h | v).
"""
function var_h_from_v(rbm::AbstractRBM, v::AbstractArray; β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_var(hidden(rbm), inputs; β)
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
    ϵ = mean(abs.(v .- v1); dims = 1:ndims(visible(rbm)))
    if ndims(v) == ndims(visible(rbm))
        return only(ϵ)
    else
        return reshape(ϵ, batch_size(visible(rbm), v))
    end
end

"""
    ∂free_energy(rbm, v)

Gradient of `free_energy(rbm, v)` with respect to model parameters.
If `v` consists of multiple samples (batches), then an average is taken.
"""
function ∂free_energy(
    rbm::AbstractRBM, v::AbstractArray;
    wts = nothing, stats = suffstats(rbm, v; wts)
)
    inputs = inputs_v_to_h(rbm, v)
    ∂v = ∂energy(visible(rbm), stats)
    ∂Γ = ∂free_energies(hidden(rbm), inputs)
    ∂h = map(∂Γ) do ∂f
        batchmean(hidden(rbm), ∂f; wts)
    end
    h = grad2mean(hidden(rbm), ∂Γ)
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

suffstats(rbm::AbstractRBM, v::AbstractArray; wts=nothing) = suffstats(visible(rbm), v; wts)

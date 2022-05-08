"""
    RBM{V,H,W}

RBM, with visible layer of type `V`, hidden layer of type `H`, and weights of type `W`.
"""
struct RBM{V,H,W}
    visible::V
    hidden::H
    w::W
    """
        RBM(visible, hidden, w)

    Creates a Restricted Boltzmann machine with `visible` and `hidden` layers and weights `w`.
    """
    function RBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray)
        @assert size(w) == (size(visible)..., size(hidden)...)
        return new{typeof(visible), typeof(hidden), typeof(w)}(visible, hidden, w)
    end
end

visible(rbm::RBM) = rbm.visible
hidden(rbm::RBM) = rbm.hidden
weights(rbm::RBM) = rbm.w
flat_w(rbm::RBM) = reshape(weights(rbm), length(visible(rbm)), length(hidden(rbm)))
flat_v(rbm::RBM, v::AbstractArray) = flatten(visible(rbm), v)
flat_h(rbm::RBM, h::AbstractArray) = flatten(hidden(rbm), h)

"""
    inputs_v_to_h(rbm, v)

Interaction inputs from visible to hidden layer.
"""
function inputs_v_to_h(rbm::RBM, v::AbstractArray)
    wflat = flat_w(rbm)
    vflat = activations_convert_maybe(wflat, flat_v(rbm, v))
    iflat = wflat' * vflat
    return reshape(iflat, size(hidden(rbm))..., batch_size(visible(rbm), v)...)
end

"""
    inputs_h_to_v(rbm, h)

Interaction inputs from hidden to visible layer.
"""
function inputs_h_to_v(rbm::RBM, h::AbstractArray)
    wflat = flat_w(rbm)
    hflat = activations_convert_maybe(wflat, flat_h(rbm, h))
    iflat = wflat * hflat
    return reshape(iflat, size(visible(rbm))..., batch_size(hidden(rbm), h)...)
end

"""
    free_energy(rbm, v)

Free energy of visible configuration (after marginalizing hidden configurations).
"""
function free_energy(rbm::RBM, v::AbstractArray)
    E = energy(visible(rbm), v)
    inputs = inputs_v_to_h(rbm, v)
    F = free_energy(hidden(rbm), inputs)
    return E + F
end

"""
    energy(rbm, v, h)

Energy of the rbm in the configuration `(v,h)`.
"""
function energy(rbm::RBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(visible(rbm), v)
    Eh = energy(hidden(rbm), h)
    Ew = interaction_energy(rbm, v, h)
    return Ev .+ Eh .+ Ew
end

"""
    interaction_energy(rbm, v, h)

Weight mediated interaction energy.
"""
function interaction_energy(rbm::RBM, v::AbstractArray, h::AbstractArray)
    bsz = batch_size(rbm, v, h)
    if ndims(visible(rbm)) == ndims(v) || ndims(hidden(rbm)) == ndims(h)
        E = -flat_v(rbm, v)' * flat_w(rbm) * flat_h(rbm, h)
    elseif length(visible(rbm)) ≥ length(hidden(rbm))
        inputs = inputs_v_to_h(rbm, v)
        E = -sum(inputs .* h; dims = 1:ndims(hidden(rbm)))
    else
        inputs = inputs_h_to_v(rbm, h)
        E = -sum(v .* inputs; dims=1:ndims(visible(rbm)))
    end
    return reshape_maybe(E, bsz)
end

"""
    sample_h_from_v(rbm, v)

Samples a hidden configuration conditional on the visible configuration `v`.
"""
function sample_h_from_v(rbm::RBM, v::AbstractArray)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_sample(hidden(rbm), inputs)
end

"""
    sample_v_from_h(rbm, h)

Samples a visible configuration conditional on the hidden configuration `h`.
"""
function sample_v_from_h(rbm::RBM, h::AbstractArray)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_sample(visible(rbm), inputs)
end

"""
    sample_v_from_v(rbm, v; steps = 1)

Samples a visible configuration conditional on another visible configuration `v`.
Ensures type stability by requiring that the returned array is of the same type as `v`.
"""
function sample_v_from_v(rbm::RBM, v::AbstractArray; steps::Int = 1)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    for _ in 1:steps
        v = oftype(v, sample_v_from_v_once(rbm, v))
    end
    return v
end

"""
    sample_h_from_h(rbm, h; steps = 1)

Samples a hidden configuration conditional on another hidden configuration `h`.
Ensures type stability by requiring that the returned array is of the same type as `h`.
"""
function sample_h_from_h(rbm::RBM, h::AbstractArray; steps::Int = 1)
    @assert size(hidden(rbm)) == size(h)[1:ndims(hidden(rbm))]
    for _ in 1:steps
        h = oftype(h, sample_h_from_h_once(rbm, h))
    end
    return h
end

function sample_v_from_v_once(rbm::RBM, v::AbstractArray)
    h = sample_h_from_v(rbm, v)
    v = sample_v_from_h(rbm, h)
    return v
end

function sample_h_from_h_once(rbm::RBM, h::AbstractArray)
    v = sample_v_from_h(rbm, h)
    h = sample_h_from_v(rbm, v)
    return h
end

"""
    mean_h_from_v(rbm, v)

Mean unit activation values, conditioned on the other layer, <h | v>.
"""
function mean_h_from_v(rbm::RBM, v::AbstractArray)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_mean(hidden(rbm), inputs)
end

"""
    mean_v_from_h(rbm, v)

Mean unit activation values, conditioned on the other layer, <v | h>.
"""
function mean_v_from_h(rbm::RBM, h::AbstractArray)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_mean(visible(rbm), inputs)
end

"""
    var_v_from_h(rbm, v)

Variance of unit activation values, conditioned on the other layer, var(v | h).
"""
function var_v_from_h(rbm::RBM, h::AbstractArray)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_var(visible(rbm), inputs)
end

"""
    var_h_from_v(rbm, v)

Variance of unit activation values, conditioned on the other layer, var(h | v).
"""
function var_h_from_v(rbm::RBM, v::AbstractArray)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_var(hidden(rbm), inputs)
end

"""
    mode_v_from_h(rbm, h)

Mode unit activations, conditioned on the other layer.
"""
function mode_v_from_h(rbm::RBM, h::AbstractArray)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_mode(visible(rbm), inputs)
end

"""
    mode_h_from_v(rbm, v)

Mode unit activations, conditioned on the other layer.
"""
function mode_h_from_v(rbm::RBM, v::AbstractArray)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_mode(hidden(rbm), inputs)
end

"""
    batch_size(rbm, v, h)

Returns the batch size if `energy(rbm, v, h)` were computed.
"""
function batch_size(rbm::RBM, v::AbstractArray, h::AbstractArray)
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
    reconstruction_error(rbm, v; steps = 1)

Stochastic reconstruction error of `v`.
"""
function reconstruction_error(rbm::RBM, v::AbstractArray; steps::Int=1)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    v1 = sample_v_from_v(rbm, v; steps)
    ϵ = mean(abs.(v .- v1); dims = 1:ndims(visible(rbm)))
    if ndims(v) == ndims(visible(rbm))
        return only(ϵ)
    else
        return reshape(ϵ, batch_size(visible(rbm), v))
    end
end

"""
    mirror(rbm)

Returns a new RBM with viible and hidden layers flipped.
"""
function mirror(rbm::RBM)
    p(i::Int) = i ≤ ndims(visible(rbm)) ? i + ndims(hidden(rbm)) : i - ndims(visible(rbm))
    perm = ntuple(p, ndims(weights(rbm)))
    w = permutedims(weights(rbm), perm)
    return RBM(hidden(rbm), visible(rbm), w)
end

"""
    ∂free_energy(rbm, v)

Gradient of `free_energy(rbm, v)` with respect to model parameters.
If `v` consists of multiple samples (batches), then an average is taken.
"""
function ∂free_energy(
    rbm::RBM, v::AbstractArray; wts = nothing, stats = suffstats(rbm, v; wts)
)
    inputs = inputs_v_to_h(rbm, v)
    ∂v = ∂energy(visible(rbm), stats)
    ∂Γ = ∂free_energies(hidden(rbm), inputs)
    ∂h = map(∂Γ) do ∂f
        batchmean(hidden(rbm), ∂f; wts)
    end
    h = grad2ave(hidden(rbm), ∂Γ)
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

function ∂interaction_energy(rbm::RBM, v::AbstractArray, h::AbstractArray; wts = nothing)
    bsz = batch_size(rbm, v, h)
    if ndims(visible(rbm)) == ndims(v) && ndims(hidden(rbm)) == ndims(h)
        wts::Nothing
        ∂wflat = -vec(v) * vec(h)'
    elseif ndims(visible(rbm)) == ndims(v)
        ∂wflat = -vec(v) * vec(batchmean(hidden(rbm), h; wts))'
    elseif ndims(hidden(rbm)) == ndims(h)
        ∂wflat = -vec(batchmean(visible(rbm), v; wts)) * vec(h)'
    else
        hflat = flatten(hidden(rbm), h)
        vflat = activations_convert_maybe(hflat, flatten(visible(rbm), v))
        @assert isnothing(wts) || size(wts) == batch_size(visible(rbm), v)
        if isnothing(wts)
            ∂wflat = -vflat * hflat' / size(vflat, 2)
        else
            @assert size(wts) == bsz
            @assert batch_size(visible(rbm), v) == batch_size(hidden(rbm), h) == size(wts)
            ∂wflat = -vflat * Diagonal(vec(wts)) * hflat' / sum(wts)
        end
    end
    ∂w = reshape(∂wflat, size(weights(rbm)))
    return ∂w
end

suffstats(rbm::RBM, v::AbstractArray; wts=nothing) = suffstats(visible(rbm), v; wts)

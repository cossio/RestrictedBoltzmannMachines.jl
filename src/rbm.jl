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

flat_w(rbm) = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
flat_v(rbm, v) = flatten(rbm.visible, v)
flat_h(rbm, h) = flatten(rbm.hidden, h)

"""
    inputs_h_from_v(rbm, v)

Interaction inputs from visible to hidden layer.
"""
function inputs_h_from_v(rbm, v)
    wflat = flat_w(rbm)
    vflat = with_eltype_of(wflat, flat_v(rbm, v))
    iflat = wflat' * vflat
    return reshape(iflat, size(rbm.hidden)..., batch_size(rbm.visible, v)...)
end

"""
    inputs_v_from_h(rbm, h)

Interaction inputs from hidden to visible layer.
"""
function inputs_v_from_h(rbm, h)
    wflat = flat_w(rbm)
    hflat = with_eltype_of(wflat, flat_h(rbm, h))
    iflat = wflat * hflat
    return reshape(iflat, size(rbm.visible)..., batch_size(rbm.hidden, h)...)
end

"""
    free_energy(rbm, v)

Free energy of visible configuration (after marginalizing hidden configurations).
"""
function free_energy(rbm, v)
    E = energy(rbm.visible, v)
    Γ = hidden_cgf(rbm, v)
    return E - Γ
end

function hidden_cgf(rbm, v)
    inputs = inputs_h_from_v(rbm, v)
    return cgf(rbm.hidden, inputs)
end

"""
    energy(rbm, v, h)

Energy of the rbm in the configuration `(v,h)`.
"""
function energy(rbm, v, h)
    Ev = energy(rbm.visible, v)
    Eh = energy(rbm.hidden, h)
    Ew = interaction_energy(rbm, v, h)
    return Ev .+ Eh .+ Ew
end

"""
    interaction_energy(rbm, v, h)

Weight mediated interaction energy.
"""
function interaction_energy(rbm, v, h)
    bsz = batch_size(rbm, v, h)
    if ndims(rbm.visible) == ndims(v) || ndims(rbm.hidden) == ndims(h)
        w_flat = flat_w(rbm)
        v_flat = with_eltype_of(w_flat, flat_v(rbm, v))
        h_flat = with_eltype_of(w_flat, flat_h(rbm, h))
        E = -v_flat' * w_flat * with_eltype_of(w_flat, flat_h(rbm, h))
    elseif length(rbm.visible) ≥ length(rbm.hidden)
        inputs = inputs_h_from_v(rbm, v)
        E = -sum(inputs .* h; dims = 1:ndims(rbm.hidden))
    else
        inputs = inputs_v_from_h(rbm, h)
        E = -sum(v .* inputs; dims=1:ndims(rbm.visible))
    end
    return reshape_maybe(E, bsz)
end

"""
    sample_h_from_v(rbm, v)

Samples a hidden configuration conditional on the visible configuration `v`.
"""
function sample_h_from_v(rbm, v)
    inputs = inputs_h_from_v(rbm, v)
    return sample_from_inputs(rbm.hidden, inputs)
end

"""
    sample_v_from_h(rbm, h)

Samples a visible configuration conditional on the hidden configuration `h`.
"""
function sample_v_from_h(rbm, h)
    inputs = inputs_v_from_h(rbm, h)
    return sample_from_inputs(rbm.visible, inputs)
end

"""
    sample_v_from_v(rbm, v; steps=1)

Samples a visible configuration conditional on another visible configuration `v`.
Ensures type stability by requiring that the returned array is of the same type as `v`.
"""
function sample_v_from_v(rbm, v; steps=1)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    for _ in 1:steps
        v = oftype(v, sample_v_from_v_once(rbm, v))
    end
    return v
end

"""
    sample_h_from_h(rbm, h; steps=1)

Samples a hidden configuration conditional on another hidden configuration `h`.
Ensures type stability by requiring that the returned array is of the same type as `h`.
"""
function sample_h_from_h(rbm, h; steps=1)
    @assert size(rbm.hidden) == size(h)[1:ndims(rbm.hidden)]
    for _ in 1:steps
        h = oftype(h, sample_h_from_h_once(rbm, h))
    end
    return h
end

function sample_v_from_v_once(rbm, v)
    h = sample_h_from_v(rbm, v)
    v = sample_v_from_h(rbm, h)
    return v
end

function sample_h_from_h_once(rbm, h)
    v = sample_v_from_h(rbm, h)
    h = sample_h_from_v(rbm, v)
    return h
end

"""
    mean_h_from_v(rbm, v)

Mean unit activation values, conditioned on the other layer, <h | v>.
"""
function mean_h_from_v(rbm, v)
    inputs = inputs_h_from_v(rbm, v)
    return mean_from_inputs(rbm.hidden, inputs)
end

"""
    mean_v_from_h(rbm, v)

Mean unit activation values, conditioned on the other layer, <v | h>.
"""
function mean_v_from_h(rbm, h)
    inputs = inputs_v_from_h(rbm, h)
    return mean_from_inputs(rbm.visible, inputs)
end

"""
    var_v_from_h(rbm, v)

Variance of unit activation values, conditioned on the other layer, var(v | h).
"""
function var_v_from_h(rbm, h)
    inputs = inputs_v_from_h(rbm, h)
    return var_from_inputs(rbm.visible, inputs)
end

"""
    var_h_from_v(rbm, v)

Variance of unit activation values, conditioned on the other layer, var(h | v).
"""
function var_h_from_v(rbm, v)
    inputs = inputs_h_from_v(rbm, v)
    return var_from_inputs(rbm.hidden, inputs)
end

"""
    mode_v_from_h(rbm, h)

Mode unit activations, conditioned on the other layer.
"""
function mode_v_from_h(rbm, h)
    inputs = inputs_v_from_h(rbm, h)
    return mode_from_inputs(rbm.visible, inputs)
end

"""
    mode_h_from_v(rbm, v)

Mode unit activations, conditioned on the other layer.
"""
function mode_h_from_v(rbm, v)
    inputs = inputs_h_from_v(rbm, v)
    return mode_from_inputs(rbm.hidden, inputs)
end

"""
    batch_size(rbm, v, h)

Returns the batch size if `energy(rbm, v, h)` were computed.
"""
function batch_size(rbm, v, h)
    v_bsz = batch_size(rbm.visible, v)
    h_bsz = batch_size(rbm.hidden, h)
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
function reconstruction_error(rbm, v; steps=1)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    v1 = sample_v_from_v(rbm, v; steps)
    ϵ = mean(abs.(v .- v1); dims = 1:ndims(rbm.visible))
    if ndims(v) == ndims(rbm.visible)
        return only(ϵ)
    else
        return reshape(ϵ, batch_size(rbm.visible, v))
    end
end

"""
    mirror(rbm)

Returns a new RBM with visible and hidden layers flipped.
"""
function mirror(rbm)
    p(i::Int) = i ≤ ndims(rbm.visible) ? i + ndims(rbm.hidden) : i - ndims(rbm.visible)
    perm = ntuple(p, ndims(rbm.w))
    w = permutedims(rbm.w, perm)
    return RBM(rbm.hidden, rbm.visible, w)
end

"""
    ∂free_energy(rbm, v)

Gradient of `free_energy(rbm, v)` with respect to model parameters.
If `v` consists of multiple samples (batches), then an average is taken.
"""
function ∂free_energy(
    rbm::RBM, v::AbstractArray; wts = nothing,
    moments = moments_from_samples(rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(rbm, v)
    ∂v = ∂energy_from_moments(rbm.visible, moments)
    ∂Γ = ∂cgfs(rbm.hidden, inputs)
    h = grad2ave(rbm.hidden, ∂Γ)
    ∂h = reshape(wmean(-∂Γ; wts, dims = (ndims(rbm.hidden.par) + 1):ndims(∂Γ)), size(rbm.hidden.par))
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return ∂RBM(∂v, ∂h, ∂w)
end

function ∂interaction_energy(rbm, v, h; wts=nothing)
    bsz = batch_size(rbm, v, h)
    if ndims(rbm.visible) == ndims(v) && ndims(rbm.hidden) == ndims(h)
        wts::Nothing
        ∂wflat = -vec(v) * vec(h)'
    elseif ndims(rbm.visible) == ndims(v)
        ∂wflat = -vec(v) * vec(batchmean(rbm.hidden, h; wts))'
    elseif ndims(rbm.hidden) == ndims(h)
        ∂wflat = -vec(batchmean(rbm.visible, v; wts)) * vec(h)'
    else
        hflat = flatten(rbm.hidden, h)
        vflat = with_eltype_of(hflat, flatten(rbm.visible, v))
        @assert isnothing(wts) || size(wts) == batch_size(rbm.visible, v)
        if isnothing(wts)
            ∂wflat = -vflat * hflat' / size(vflat, 2)
        else
            @assert size(wts) == bsz
            @assert batch_size(rbm.visible, v) == batch_size(rbm.hidden, h) == size(wts)
            ∂wflat = -vflat * Diagonal(vec(wts)) * hflat' / sum(wts)
        end
    end
    ∂w = reshape(∂wflat, size(rbm.w))
    return ∂w
end

"""
    RBM{V,H,W}

RBM, with visible layer of type `V`, hidden layer of type `H`, and weights of type `W`.
"""
struct RBM{V, H, W}
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

function _validate_layer_parameters(rbm)
    _validate_layer_parameters(rbm.visible)
    _validate_layer_parameters(rbm.hidden)
    return nothing
end

flat_w(rbm) = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
flat_v(rbm, v) = flatten(rbm.visible, v)
flat_h(rbm, h) = flatten(rbm.hidden, h)

# reshape a visible-sized (resp. hidden-sized) array so it broadcasts along the
# visible (resp. hidden) dimensions of `rbm.w`
_along_visible(rbm, x::AbstractArray) = reshape(x, size(rbm.visible)..., map(one, size(rbm.hidden))...)
_along_hidden(rbm, x::AbstractArray) = reshape(x, map(one, size(rbm.visible))..., size(rbm.hidden)...)

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
free_energy(rbm, v) = energy(rbm.visible, v) - hidden_cgf(rbm, v)
free_energy_v(rbm, v) = free_energy(rbm, v)
free_energy_h(rbm, h) = energy(rbm.hidden, h) - visible_cgf(rbm, h)

hidden_cgf(rbm, v) = cgf(rbm.hidden, inputs_h_from_v(rbm, v))
visible_cgf(rbm, h) = cgf(rbm.visible, inputs_v_from_h(rbm, h))

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
    if ndims(rbm.visible) == ndims(v)
        # v is a single sample (h single or batched): v'*w*h as matrix products
        w_flat = flat_w(rbm)
        v_flat = with_eltype_of(w_flat, flat_v(rbm, v))
        h_flat = with_eltype_of(w_flat, flat_h(rbm, h))
        E = -(v_flat' * w_flat * h_flat)
    elseif ndims(rbm.hidden) == ndims(h)
        # v is batched, h is single: compute w*h first (small), then v'*(w*h)
        # This avoids converting the large batched v array to match w's eltype.
        w_flat = flat_w(rbm)
        h_flat = with_eltype_of(w_flat, flat_h(rbm, h))
        wh = w_flat * h_flat
        v_flat = flat_v(rbm, v)
        E = -(v_flat' * wh)
    elseif length(rbm.visible) ≥ length(rbm.hidden)
        inputs = inputs_h_from_v(rbm, v)
        E = -sum(inputs .* h; dims = 1:ndims(rbm.hidden))
    else
        inputs = inputs_v_from_h(rbm, h)
        E = -sum(v .* inputs; dims = 1:ndims(rbm.visible))
    end
    return reshape_maybe(E, bsz)
end

"""
    sample_h_from_v(rbm, v)

Samples a hidden configuration conditional on the visible configuration `v`.
"""
sample_h_from_v(rbm, v) = sample_from_inputs(rbm.hidden, inputs_h_from_v(rbm, v))

"""
    sample_v_from_h(rbm, h)

Samples a visible configuration conditional on the hidden configuration `h`.
"""
sample_v_from_h(rbm, h) = sample_from_inputs(rbm.visible, inputs_v_from_h(rbm, h))

"""
    sample_v_from_v(rbm, v; steps=1)

Samples a visible configuration conditional on another visible configuration `v`.
Ensures type stability by requiring that the returned array is of the same type as `v`.
"""
function sample_v_from_v(rbm, v; steps = 1)
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
function sample_h_from_h(rbm, h; steps = 1)
    @assert size(rbm.hidden) == size(h)[1:ndims(rbm.hidden)]
    for _ in 1:steps
        h = oftype(h, sample_h_from_h_once(rbm, h))
    end
    return h
end

sample_v_from_v_once(rbm, v) = sample_v_from_h(rbm, sample_h_from_v(rbm, v))
sample_h_from_h_once(rbm, h) = sample_h_from_v(rbm, sample_v_from_h(rbm, h))

"""
    mean_h_from_v(rbm, v)

Mean unit activation values, conditioned on the other layer, <h | v>.
"""
mean_h_from_v(rbm, v) = mean_from_inputs(rbm.hidden, inputs_h_from_v(rbm, v))

"""
    mean_v_from_h(rbm, h)

Mean unit activation values, conditioned on the other layer, <v | h>.
"""
mean_v_from_h(rbm, h) = mean_from_inputs(rbm.visible, inputs_v_from_h(rbm, h))

"""
    var_v_from_h(rbm, h)

Variance of unit activation values, conditioned on the other layer, var(v | h).
"""
var_v_from_h(rbm, h) = var_from_inputs(rbm.visible, inputs_v_from_h(rbm, h))

"""
    var_h_from_v(rbm, v)

Variance of unit activation values, conditioned on the other layer, var(h | v).
"""
var_h_from_v(rbm, v) = var_from_inputs(rbm.hidden, inputs_h_from_v(rbm, v))

"""
    mode_v_from_h(rbm, h)

Mode unit activations, conditioned on the other layer.
"""
mode_v_from_h(rbm, h) = mode_from_inputs(rbm.visible, inputs_v_from_h(rbm, h))

"""
    mode_h_from_v(rbm, v)

Mode unit activations, conditioned on the other layer.
"""
mode_h_from_v(rbm, v) = mode_from_inputs(rbm.hidden, inputs_h_from_v(rbm, v))

"""
    batch_size(rbm, v, h)

Returns the batch size if `energy(rbm, v, h)` were computed.
"""
batch_size(rbm, v, h) = join_batch_size(batch_size(rbm.visible, v), batch_size(rbm.hidden, h))

# broadcast-style join of two batch sizes (either may be empty); at most one tail is nonempty
function join_batch_size(bsz_1::Dims, bsz_2::Dims)
    D = min(length(bsz_1), length(bsz_2))
    head = ntuple(D) do d
        bmin, bmax = minmax(bsz_1[d], bsz_2[d])
        @assert bmin == 1 || bmin == bmax
        bmax
    end
    return (head..., bsz_1[(D + 1):end]..., bsz_2[(D + 1):end]...)
end

"""
    reconstruction_error(rbm, v; steps = 1)

Stochastic reconstruction error of `v`.
"""
function reconstruction_error(rbm, v; steps = 1)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    v1 = sample_v_from_v(rbm, v; steps)
    ϵ = mean(abs.(v .- v1); dims = 1:ndims(rbm.visible))
    return reshape_maybe(ϵ, batch_size(rbm.visible, v))
end

"""
    mirror(rbm)

Returns a new RBM with visible and hidden layers flipped.
"""
function mirror(rbm)
    perm = ntuple(Val(ndims(rbm.w))) do i
        if i ≤ ndims(rbm.hidden)
            i + ndims(rbm.visible)
        else
            i - ndims(rbm.hidden)
        end
    end
    w = permutedims(rbm.w, perm)
    return RBM(rbm.hidden, rbm.visible, w)
end

"""
    potts_to_gumbel(rbm)

Converts Potts layers to PottsGumbel layers.
"""
function potts_to_gumbel(rbm::RBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

"""
    gumbel_to_potts(rbm)

Converts PottsGumbel layers to Potts layers.
"""
function gumbel_to_potts(rbm::RBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

# total_{mean,var,meanvar}_{h_from_v,v_from_h}: batch-averaged conditional statistics
for (stat, what) in ((:mean, "mean"), (:var, "variance"), (:meanvar, "mean and variance"))
    from_inputs = Symbol(:total_, stat, :_from_inputs)
    h_from_v = Symbol(:total_, stat, :_h_from_v)
    v_from_h = Symbol(:total_, stat, :_v_from_h)
    @eval begin
        """
            $($h_from_v)(rbm, v; [wts])

        Total $($what) of hidden unit activations given the visible activities `v`,
        averaged over the batch with weights `wts`.
        """
        function $h_from_v(rbm, v::AbstractArray; wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, v))
            return $from_inputs(rbm.hidden, inputs_h_from_v(rbm, v); wts)
        end

        """
            $($v_from_h)(rbm, h; [wts])

        Total $($what) of visible unit activations given the hidden activities `h`,
        averaged over the batch with weights `wts`.
        """
        function $v_from_h(rbm, h::AbstractArray; wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, h))
            return $from_inputs(rbm.visible, inputs_v_from_h(rbm, h); wts)
        end
    end
end

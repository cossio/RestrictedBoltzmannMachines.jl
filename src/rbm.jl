"""
    RBM

Represents a restricted Boltzmann Machine.
"""
struct RBM{V<:AbstractLayer, H<:AbstractLayer, W<:AbstractArray}
    visible::V
    hidden::H
    w::W
    function RBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray)
        @assert size(w) == (size(visible)..., size(hidden)...)
        return new{typeof(visible), typeof(hidden), typeof(w)}(visible, hidden, w)
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
    return Ev .+ Eh .+ Ew
end

"""
    interaction_energy(rbm, v, h)

Weight mediated interaction energy.
"""
function interaction_energy(rbm::RBM, v::AbstractArray, h::AbstractArray)
    @assert size(v)[1:ndims(rbm.visible)] == size(rbm.visible)
    @assert size(h)[1:ndims(rbm.hidden)]  == size(rbm.hidden)
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    if ndims(rbm.visible) == ndims(v) && ndims(rbm.hidden) == ndims(h)
        return -vec(v)' * wmat * vec(h)
    elseif ndims(rbm.visible) < ndims(v) && ndims(rbm.hidden) == ndims(h)
        vflat = reshape(v, length(rbm.visible), :)
        Eflat = -vec(h)' * wmat' * vflat
        return reshape(Eflat, size(v)[(ndims(rbm.visible) + 1):end])
    elseif ndims(rbm.visible) == ndims(v) && ndims(rbm.hidden) < ndims(h)
        hflat = reshape(h, length(rbm.hidden), :)
        Eflat = -vec(v)' * wmat * hflat
        return reshape(Eflat, size(h)[(ndims(rbm.hidden) + 1):end])
    else
        @assert size(v)[(ndims(rbm.visible) + 1):end] == size(h)[(ndims(rbm.hidden) + 1):end]
        vflat = reshape(v, length(rbm.visible), :)
        hflat = reshape(h, length(rbm.hidden), :)
        if size(vflat, 1) ≥ size(hflat, 1)
            Eflat = -dropdims(sum((wmat' * vflat) .* hflat; dims=1); dims=1)
        else
            Eflat = -dropdims(sum(vflat .* (wmat * hflat); dims=1); dims=1)
        end
        return reshape(Eflat, size(v)[(ndims(rbm.visible) + 1):end])
    end
end

"""
    inputs_v_to_h(rbm, v)

Interaction inputs from visible to hidden layer.
"""
function inputs_v_to_h(rbm::RBM, v::AbstractArray)
    @assert size(v)[1:ndims(rbm.visible)] == size(rbm.visible)
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    vflat = reshape(v, length(rbm.visible), :)
    vconv = activations_convert_maybe(wmat, vflat)
    iflat = wmat' * vconv
    return reshape(iflat, size(rbm.hidden)..., size(v)[(ndims(rbm.visible) + 1):end]...)
end

"""
    inputs_h_to_v(rbm, h)

Interaction inputs from hidden to visible layer.
"""
function inputs_h_to_v(rbm::RBM, h::AbstractArray)
    @assert size(h)[1:ndims(rbm.hidden)] == size(rbm.hidden)
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    hflat = reshape(h, length(rbm.hidden), :)
    hconv = activations_convert_maybe(wmat, hflat)
    iflat = wmat * hconv
    return reshape(iflat, size(rbm.visible)..., size(h)[(ndims(rbm.hidden) + 1):end]...)
end

# convert to common eltype before matrix multiply, to make sure we hit BLAS
activations_convert_maybe(::AbstractArray{W}, x::AbstractArray{X}) where {W,X} = map(W, x)
activations_convert_maybe(::AbstractArray{T}, x::AbstractArray{T}) where {T} = x

"""
    free_energy(rbm, v; β = 1)

Free energy of visible configuration (after marginalizing hidden configurations).
"""
function free_energy(rbm::RBM, v::AbstractArray; β::Real = true)
    E = energy(rbm.visible, v)::Union{Number, AbstractVector}
    inputs = inputs_v_to_h(rbm, v)::AbstractArray
    F = free_energy(rbm.hidden, inputs; β)::Union{Number, AbstractVector}
    return E + F
end

"""
    sample_h_from_v(rbm, v; β = 1)

Samples a hidden configuration conditional on the visible configuration `v`.
"""
function sample_h_from_v(rbm::RBM, v::AbstractArray; β::Real = true)
    inputs = inputs_v_to_h(rbm, v)
    return transfer_sample(rbm.hidden, inputs; β)
end

"""
    sample_v_from_h(rbm, h; β = 1)

Samples a visible configuration conditional on the hidden configuration `h`.
"""
function sample_v_from_h(rbm::RBM, h::AbstractArray; β::Real = true)
    inputs = inputs_h_to_v(rbm, h)
    return transfer_sample(rbm.visible, inputs; β)
end

"""
    sample_v_from_v(rbm, v; β = 1, steps = 1)

Samples a visible configuration conditional on another visible configuration `v`.
"""
function sample_v_from_v(rbm::RBM, v::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    v_ = copy(v)
    for _ in 1:steps
        v_ .= sample_v_from_v_once(rbm, v_; β)
    end
    return v_
end

"""
    sample_h_from_h(rbm, h; β = 1, steps = 1)

Samples a hidden configuration conditional on another hidden configuration `h`.
"""
function sample_h_from_h(rbm::RBM, h::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(rbm.hidden) == size(h)[1:ndims(rbm.hidden)]
    h_ = copy(h)
    for _ in 1:steps
        h_ .= sample_h_from_h_once(rbm, h_; β)
    end
    return h_
end

function sample_v_from_v_once(rbm::RBM, v::AbstractArray; β::Real = true)
    h = sample_h_from_v(rbm, v; β)
    v = sample_v_from_h(rbm, h; β)
    return v
end

function sample_h_from_h_once(rbm::RBM, h::AbstractArray; β::Real = true)
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

"""
    reconstruction_error(rbm, v; β = 1, steps = 1)

Stochastic reconstruction error of `v`.
"""
function reconstruction_error(rbm::RBM, v::AbstractTensor; β::Real = true, steps::Int = 1)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    v_ = sample_v_from_v(rbm, v; β, steps)
    ϵ = mean(abs.(v .- v_); dims = 1:ndims(rbm.visible))
    if ndims(v) == ndims(rbm.visible)
        return only(ϵ)
    else
        return reshape(ϵ, size(v)[end])
    end
end

"""
    mirror(rbm)

Returns a new RBM with viible and hidden layers flipped.
"""
function mirror(rbm::RBM)
    function p(i)
        if i ≤ ndims(rbm.visible)
            return i + ndims(rbm.hidden)
        else
            return i - ndims(rbm.visible)
        end
    end
    perm = ntuple(p, ndims(rbm.w))
    w = permutedims(rbm.w, perm)
    return RBM(rbm.hidden, rbm.visible, w)
end

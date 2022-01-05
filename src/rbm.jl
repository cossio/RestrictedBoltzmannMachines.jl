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
function energy(rbm::RBM, v::AbstractTensor, h::AbstractTensor)
    check_size(rbm, v, h)
    Ev::Union{Number, AbstractVector} = energy(rbm.visible, v)
    Eh::Union{Number, AbstractVector} = energy(rbm.hidden, h)
    Ew::Union{Number, AbstractVector} = interaction_energy(rbm, v, h)
    return (Ev .+ Eh .+ Ew)::Union{Number, AbstractVector}
end

"""
    interaction_energy(rbm, v, h)

Weight mediated interaction energy.
"""
function interaction_energy(rbm::RBM, v::AbstractArray, h::AbstractArray)
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    vflat = flatten(rbm.visible, v)
    hflat = flatten(rbm.hidden, h)
    return flat_interaction_energy(wmat, vflat, hflat)
end

function flat_interaction_energy(w::AbstractMatrix, v::AbstractVector, h::AbstractVector)
    @assert size(w) == (length(v), length(h))
    return -dot(v, w, h)::Number
end

function flat_interaction_energy(w::AbstractMatrix, v::AbstractMatrix, h::AbstractVector)
    @assert size(w) == (size(v, 1), size(h, 1))
    E::AbstractVector = -v' * (w * h)
    return E
end

function flat_interaction_energy(w::AbstractMatrix, v::AbstractVector, h::AbstractMatrix)
    @assert size(w) == (size(v, 1), size(h, 1))
    E::AbstractVector = -h' * (w' * v)
    return E
end

function flat_interaction_energy(w::AbstractMatrix, v::AbstractMatrix, h::AbstractMatrix)
    @assert size(w) == (size(v, 1), size(h, 1))
    @assert size(v, 2) == size(h, 2) # batch size
    if size(v, 1) ≥ size(h, 1)
        E = -sum_((w' * v) .* h; dims=1)
    else
        E = -sum_(v .* (w * h); dims=1)
    end
    @assert length(E) == size(v, 2) == size(h, 2)
    return E::AbstractVector
end

"""
    inputs_v_to_h(rbm, v)

Interaction inputs from visible to hidden layer.
"""
function inputs_v_to_h(rbm::RBM, v::AbstractArray)
    vflat = flatten(rbm.visible, v)
    wflat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    vconv = activations_convert_maybe(wflat, vflat)
    return unflatten(rbm.hidden, wflat' * vconv)
end

"""
    inputs_h_to_v(rbm, h)

Interaction inputs from hidden to visible layer.
"""
function inputs_h_to_v(rbm::RBM, h::AbstractArray)
    hflat = flatten(rbm.hidden, h)
    wflat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    hconv = activations_convert_maybe(wflat, hflat)
    return unflatten(rbm.visible, wflat * hconv)
end

# convert to common eltype before matrix multiply, to make sure we hit BLAS
activations_convert_maybe(w::AbstractArray{W}, x::AbstractArray{X}) where {W,X} = map(W, x)
activations_convert_maybe(w::AbstractArray{T}, x::AbstractArray{T}) where {T} = x

"""
    free_energy(rbm, v; β = 1)

Free energy of visible configuration (after marginalizing hidden configurations).
"""
function free_energy(rbm::RBM, v::AbstractArray; β::Real = true)
    E = energy(rbm.visible, v)
    inputs = inputs_v_to_h(rbm, v)
    F = free_energy(rbm.hidden, inputs; β)
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
    check_size(rbm.visible, v)
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
function sample_h_from_h(rbm::RBM, h::AbstractArray; β::Real = true, steps::Int = 1)
    check_size(rbm.hidden, h)
    @assert steps ≥ 1
    h_ = sample_h_from_h_once(rbm, h; β)
    for _ in 1:(steps - 1)
        h_ = sample_h_from_h_once(rbm, h_; β)
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

∂w_flat(v::AbstractVector, h::AbstractVector, wts::Nothing = nothing) = -v * h'

function ∂w_flat(v::AbstractMatrix, h::AbstractMatrix, wts::Nothing = nothing)
    @assert size(v, 2) == size(h, 2)
    return -v * h' / size(v, 2)
end

function ∂w_flat(v::AbstractMatrix, h::AbstractMatrix, wts::AbstractVector)
    @assert size(v, 2) == size(h, 2) == length(wts)
    return -v * Diagonal(wts) * h' / size(v, 2)
end

function ∂free_energy(
    rbm::RBM, v::AbstractTensor;
    inputs::AbstractArray = inputs_v_to_h(rbm, v), wts::Wts = nothing
)
    check_size(rbm, v, inputs)
    h = transfer_mean(rbm.hidden, inputs)
    ∂v = ∂energy(rbm.visible, v; wts)
    ∂h = ∂free_energy(rbm.hidden, inputs; wts)

    check_size(rbm, v, h)
    v_ = flatten(rbm.visible, v)
    h_ = flatten(rbm.hidden, h)
    ∂w = ∂w_flat(v_, h_, wts)
    @assert size(∂w) == (length(rbm.visible), length(rbm.hidden))

    return (visible = ∂v, hidden = ∂h, w = reshape(∂w, size(rbm.w)))
end

"""
    reconstruction_error(rbm, v; β = 1, steps = 1)

Stochastic reconstruction error of `v`.
"""
function reconstruction_error(rbm::RBM, v::AbstractTensor; β::Real = true, steps::Int = 1)
    check_size(rbm.visible, v)
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

function check_size(
    rbm::RBM{<:AbstractLayer{N}, <:AbstractLayer{M}},
    v::AbstractTensor{N}, h::AbstractTensor{M}
) where {N, M}
    check_size(rbm.visible, v)
    check_size(rbm.hidden, h)
end

function check_size(
    rbm::RBM{<:AbstractLayer{N}, <:AbstractLayer{M}},
    v::AbstractTensor{N}, h::AbstractTensor
) where {N, M}
    check_size(rbm.visible, v)
    check_size(rbm.hidden, h)
end

function check_size(
    rbm::RBM{<:AbstractLayer{N}, <:AbstractLayer{M}},
    v::AbstractTensor, h::AbstractTensor{M}
) where {N, M}
    check_size(rbm.visible, v)
    check_size(rbm.hidden, h)
end

function check_size(rbm::RBM, v::AbstractTensor, h::AbstractTensor)
    check_size(rbm.visible, v)
    check_size(rbm.hidden, h)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    @assert size(h) == (size(rbm.hidden)..., size(h)[end])
    if size(v)[end] ≠ size(h)[end]
        throw(DimensionMismatch(
            """
            inconsistent batch dimensions; got size(v) = $(size(v)), size(h) = $(size(h))
            but size(visible) = $(size(rbm.visible)), size(hidden) = $(size(rbm.hidden))
            """
        ))
    else
        return true
    end
end

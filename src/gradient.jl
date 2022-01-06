"""
    ∂energy(layer; ts...)

Derivative of average energy of configurations with respect to layer parameters,
where `ts...` refers to the sufficient statistics from samples required by the layer.
See [`sufficient_statistics`](@ref).
"""
function ∂energy end

"""
    ∂free_energy(layer, inputs = 0; wts = 1)

Unit activation moments, conjugate to layer parameters.
These are obtained by differentiating `free_energies` with respect to the layer parameters.
Averages over configurations (weigthed by `wts`).
"""
function ∂free_energy(
    layer::AbstractLayer{N}, inputs::Union{Real, AbstractTensor{N}}; wts::Nothing = nothing
) where {N}
    check_size(layer, inputs)
    layer_eff = effective(layer, inputs)
    @assert size(layer_eff) == size(layer)
    return ∂free_energy(layer_eff)
end

function ∂free_energy(layer::AbstractLayer, inputs::AbstractTensor; wts::Wts = nothing)
    check_size(layer, inputs)
    layer_eff = effective(layer, inputs)
    ∂F = ∂free_energy(layer_eff)
    return map(∂F) do ∂fs
        @assert size(∂fs) == size(layer_eff)
        ∂ω = batch_mean(∂fs, wts)
        @assert size(∂ω) == size(layer)
        ∂ω
    end
end

function ∂free_energy(
    rbm::RBM, v::AbstractTensor; wts::Wts = nothing,
    ts = sufficient_statistics(rbm.visible, v; wts)
)
    inputs = inputs_v_to_h(rbm, v)
    h = transfer_mean(rbm.hidden, inputs)
    ∂v = ∂energy(rbm.visible; ts...)
    ∂h = ∂free_energy(rbm.hidden, inputs; wts)

    v_ = flatten(rbm.visible, activations_convert_maybe(h, v))
    h_ = flatten(rbm.hidden, h)
    ∂w = ∂w_flat(v_, h_, wts)
    @assert size(∂w) == (length(rbm.visible), length(rbm.hidden))

    return (visible = ∂v, hidden = ∂h, w = reshape(∂w, size(rbm.w)))
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

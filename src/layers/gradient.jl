"""
    ∂energies(layer, x)

Derivative of energies of configurations `x` with respect to layer parameters.
Similar to `∂energy`, but does not average over configurations.
"""
function ∂energies(layer::AbstractLayer, x::Real)
    xs = FillArrays.Fill(x, size(layer))
    return ∂energies(layer, xs)
end

"""
    ∂energy(layer, inputs = 0; β = 1, wts = 1)

Derivative of average energy of configurations with respect to layer parameters.
Similar to `∂energies`, but averages over configurations (weigthed by `wts`).
"""
∂energy(layer::AbstractLayer, x::Real; wts::Nothing = nothing) = ∂energies(layer, x)
function ∂energy(
    layer::AbstractLayer{N}, x::AbstractTensor{N}; wts::Nothing = nothing
) where {N}
    check_size(layer, x)
    @assert size(layer) == size(x)
    return ∂energies(layer, x)
end
function ∂energy(layer::AbstractLayer, x::AbstractTensor{N}; wts::Wts = nothing) where {N}
    check_size(layer, x)
    @assert size(x) == (size(layer)..., size(x, N))
    dE = ∂energies(layer, x)
    return map(dE) do dx
        @assert size(dx) == size(x)
        batch_mean(dx, wts)
    end
end

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

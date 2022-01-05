function effective(layer::AbstractLayer, input::Real; β::Real = true)
    inputs = FillArrays.Falses(size(layer))
    return effective(layer, inputs; β)
end

"""
    energy(layer, x)

Layer energy, reduced over layer dimensions.
"""
function energy(layer::AbstractLayer{N}, x::AbstractTensor{N})::Number where {N}
    check_size(layer, x)
    return sum(energies(layer, x))
end

function energy(layer::AbstractLayer, x::AbstractTensor{N})::AbstractVector where {N}
    check_size(layer, x)
    E = sum(energies(layer, x); dims = 1:ndims(layer))
    return reshape(E, size(x, N))::AbstractVector
end

function energy(layer::Union{Binary,Spin,Potts}, x::AbstractTensor)
    check_size(layer, x)
    xconv = activations_convert_maybe(layer.θ, x)
    return -flatten(layer, xconv)' * vec(layer.θ)
end

function energy(layer::AbstractLayer, x::Real)
    xs = FillArrays.Fill(x, size(layer))
    return energy(layer, xs)
end

"""
    free_energy(layer, inputs = 0; β = 1)

Cumulant generating function of layer, reduced over layer dimensions.
"""
function free_energy(
    layer::AbstractLayer{N}, inputs::AbstractTensor{N}; β::Real = true
) where {N}
    check_size(layer, inputs)
    F = free_energies(layer, inputs; β)
    return sum(F)
end

function free_energy(
    layer::AbstractLayer, inputs::AbstractTensor{N}; β::Real = true
) where {N}
    check_size(layer, inputs)
    F = free_energies(layer, inputs; β)
    f = sum(F; dims = layerdims(layer))
    return reshape(f, size(inputs, N))
end

function free_energy(layer::AbstractLayer, input::Real = false; β::Real = true)
    inputs = FillArrays.Fill(input, size(layer))
    return free_energy(layer, inputs; β)
end

"""
    transfer_sample(layer, inputs = 0; β = 1)

Samples layer configurations conditioned on inputs.
"""
function transfer_sample(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    check_size(layer, inputs)
    layer_ = effective(layer, inputs; β)
    return transfer_sample(layer_)
end

"""
    transfer_mode(layer, inputs = 0)

Mode of unit activations.
"""
function transfer_mode(layer::AbstractLayer, inputs::Union{Real, AbstractTensor})
    check_size(layer, inputs)
    layer_ = effective(layer, inputs)
    return transfer_mode(layer_)
end

"""
    transfer_mean(layer, inputs = 0; β = 1)

Mean of unit activations.
"""
function transfer_mean(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    check_size(layer, inputs)
    layer_ = effective(layer, inputs; β)
    return transfer_mean(layer_)
end

"""
    transfer_var(layer, inputs = 0; β = 1)

Variance of unit activations.
"""
function transfer_var(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    check_size(layer, inputs)
    layer_ = effective(layer, inputs; β)
    return transfer_var(layer_)
end

"""
    transfer_std(layer, inputs = 0; β = 1)

Standard deviation of unit activations.
"""
function transfer_std(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_std(layer_)
end

"""
    transfer_mean_abs(layer, inputs = 0; β = 1)

Mean of absolute value of unit activations.
"""
function transfer_mean_abs(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_mean_abs(layer_)
end

"""
    free_energies(layer, inputs = 0; β = 1)

Cumulant generating function of units in layer (not reduced over layer dimensions).
"""
function free_energies(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return free_energies(layer_) / β
end

"""
    energies(layer, x)

Energies of units in layer (not reduced over layer dimensions).
"""
function energies(layer::Union{Binary, Spin, Potts}, x::AbstractTensor)
    check_size(layer, x)
    return -layer.θ .* x
end

function energies(layer::AbstractLayer, x::Real)
    xs = FillArrays.Fill(x, size(layer))
    return energies(layer, xs)
end

const _ThetaLayers = Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}
Base.ndims(layer::_ThetaLayers) = ndims(layer.θ)
Base.size(layer::_ThetaLayers) = size(layer.θ)
Base.size(layer::_ThetaLayers, d::Int) = size(layer.θ, d)
Base.length(layer::_ThetaLayers) = length(layer.θ)

layerdims(layer) = ntuple(identity, ndims(layer))

"""
    flatten(layer, x)

Flattens `x` into a scalar, vector, or matrix (where last dimension is batch),
consistently with `layer` dimensions.
"""
flatten(::AbstractLayer, x::Real) = x
function flatten(layer::AbstractLayer{N}, x::AbstractTensor{N}) where {N}
    check_size(layer, x)
    @assert size(x) == size(layer)
    return reshape(x, length(layer))
end
function flatten(layer::AbstractLayer, x::AbstractTensor{N}) where {N}
    check_size(layer, x)
    @assert size(x) == (size(layer)..., size(x, N))
    return reshape(x, length(layer), size(x, N))
end

"""
    unflatten(layer, x)

Given a flattened (scalar, vector, or matrix) `x`, reshapes it into to match the
size of `layer`.
"""
unflatten(layer::AbstractLayer, x::Real) = FillArrays.Fill(x, size(layer))
function unflatten(layer::AbstractLayer, x::AbstractVector)
    @assert length(layer) == length(x)
    return reshape(x, size(layer))
end
function unflatten(layer::AbstractLayer, x::AbstractMatrix)
    @assert length(layer) == size(x, 1)
    return reshape(x, size(layer)..., size(x, 2))
end

function pReLU(layer::dReLU)
    γ = @. 2layer.γp * layer.γn / (layer.γp + layer.γn)
    η = @. (layer.γn - layer.γp) / (layer.γp + layer.γn)
    θ = @. (layer.θp * layer.γn + layer.θn * layer.γp) / (layer.γp + layer.γn)
    Δ = @. γ * (layer.θp - layer.θn) / (layer.γp + layer.γn)
    return pReLU(θ, γ, Δ, η)
end

function dReLU(layer::pReLU)
    γp = @. layer.γ / (1 + layer.η)
    γn = @. layer.γ / (1 - layer.η)
    θp = @. layer.θ + layer.Δ / (1 + layer.η)
    θn = @. layer.θ - layer.Δ / (1 - layer.η)
    return dReLU(θp, θn, γp, γn)
end

function xReLU(layer::dReLU)
    γ = @. 2layer.γp * layer.γn / (layer.γp + layer.γn)
    ξ = @. (layer.γn - layer.γp) / (layer.γp + layer.γn - abs(layer.γn - layer.γp))
    θ = @. (layer.θp * layer.γn + layer.θn * layer.γp) / (layer.γp + layer.γn)
    Δ = @. γ * (layer.θp - layer.θn) / (layer.γp + layer.γn)
    return xReLU(θ, γ, Δ, ξ)
end

function dReLU(layer::xReLU)
    ξp = @. (1 + abs(layer.ξ)) / (1 + max(2layer.ξ, 0))
    ξn = @. (1 + abs(layer.ξ)) / (1 - min(2layer.ξ, 0))
    γp = @. layer.γ * ξp
    γn = @. layer.γ * ξn
    θp = @. layer.θ + layer.Δ * ξp
    θn = @. layer.θ - layer.Δ * ξn
    return dReLU(θp, θn, γp, γn)
end

function xReLU(layer::pReLU)
    ξ = @. layer.η / (1 - abs(layer.η))
    return xReLU(layer.θ, layer.γ, layer.Δ, ξ)
end

function pReLU(layer::xReLU)
    η = @. layer.ξ / (1 + abs(layer.ξ))
    return pReLU(layer.θ, layer.γ, layer.Δ, η)
end

dReLU(layer::Gaussian) = dReLU(layer.θ, layer.θ, layer.γ, layer.γ)
pReLU(layer::Gaussian) = pReLU(dReLU(layer))
xReLU(layer::Gaussian) = xReLU(dReLU(layer))

number_of_colors(layer::Potts) = layer.q
number_of_colors(layer) = 1

#dReLU(layer::ReLU) = dReLU(layer.θ, zero(layer.θ), layer.γ, inf.(layer.γ))

# function pReLU(layer::ReLU)
#     θ = layer.θ
#     γ = 2layer.γ
#     η = one.(layer.γ)
#     Δ = zero.(layer.θ)
#     return pReLU(θ, γ, Δ, η)
# end

# function xReLU(layer::ReLU)
#     θ = layer.θ
#     γ = 2layer.γ
#     ξ = inf.(layer.γ)
#     Δ = zero.(layer.θ)
#     return xReLU(θ, γ, Δ, ξ)
# end

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

function check_size(layer::AbstractLayer{N}, x::AbstractTensor{N}) where {N}
    size(x) == size(layer) || throw_size_mismatch_error(layer, x)
end

function check_size(layer::AbstractLayer, x::AbstractTensor{N}) where {N}
    size(x) == (size(layer)..., size(x, N)) || throw_size_mismatch_error(layer, x)
end

check_size(::AbstractLayer, ::Real) = true

function throw_size_mismatch_error(layer::AbstractLayer, x::AbstractTensor)
    throw(DimensionMismatch(
        "size(layer)=$(size(layer)) inconsistent with size(x)=$(size(x))"
    ))
    return false
end

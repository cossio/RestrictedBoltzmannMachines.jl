abstract type AbstractLayer{N} end

Base.ndims(::AbstractLayer{N}) where {N} = N

function effective(layer::AbstractLayer, input::Real; β::Real = true)
    inputs = FillArrays.Fill(input, size(layer))
    return effective(layer, inputs; β)
end

"""
    energy(layer, x)

Layer energy, reduced over layer dimensions.
"""
function energy(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    Es = energies(layer, x)
    if ndims(layer) == ndims(x)
        return sum(Es)
    else
        E = sum(Es; dims = 1:ndims(layer))
        return reshape(E, size(x)[(ndims(layer) + 1):end])
    end
end

"""
    free_energy(layer, inputs = 0; β = 1)

Cumulant generating function of layer, reduced over layer dimensions.
"""
function free_energy(layer::AbstractLayer, inputs::AbstractArray; β::Real = true)
    F = free_energies(layer, inputs; β)
    if ndims(layer) == ndims(inputs)
        return sum(F)
    else
        f = sum(F; dims = 1:ndims(layer))
        return reshape(f, size(inputs)[(ndims(layer) + 1):end])
    end
end

function free_energy(layer::AbstractLayer, input::Real = false; β::Real = true)
    inputs = FillArrays.Fill(input, size(layer))
    return free_energy(layer, inputs; β)::Number
end

"""
    transfer_sample(layer, inputs = 0; β = 1)

Samples layer configurations conditioned on inputs.
"""
function transfer_sample(
    layer::AbstractLayer, inputs::Union{Real, AbstractArray}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_sample(layer_)
end

"""
    transfer_mode(layer, inputs = 0)

Mode of unit activations.
"""
function transfer_mode(layer::AbstractLayer, inputs::Union{Real, AbstractArray})
    layer_ = effective(layer, inputs)
    return transfer_mode(layer_)
end

"""
    transfer_mean(layer, inputs = 0; β = 1)

Mean of unit activations.
"""
function transfer_mean(
    layer::AbstractLayer, inputs::Union{Real, AbstractArray}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_mean(layer_)
end

"""
    transfer_var(layer, inputs = 0; β = 1)

Variance of unit activations.
"""
function transfer_var(
    layer::AbstractLayer, inputs::Union{Real, AbstractArray}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_var(layer_)
end

"""
    transfer_std(layer, inputs = 0; β = 1)

Standard deviation of unit activations.
"""
function transfer_std(
    layer::AbstractLayer, inputs::Union{Real, AbstractArray}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_std(layer_)
end

"""
    transfer_mean_abs(layer, inputs = 0; β = 1)

Mean of absolute value of unit activations.
"""
function transfer_mean_abs(
    layer::AbstractLayer, inputs::Union{Real, AbstractArray}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_mean_abs(layer_)
end

"""
    free_energies(layer, inputs = 0; β = 1)

Cumulant generating function of units in layer (not reduced over layer dimensions).
"""
function free_energies(
    layer::AbstractLayer, inputs::Union{Real, AbstractArray}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return free_energies(layer_) / β
end

"""
    sufficient_statistics(layer, data; [wts])

Returns a `NamedTuple` of the sufficient statistics used by the layer.
"""
function sufficient_statistics end

"""
    batchdims(layer, x)

Indices of batch dimensions in `x`, with respect to `layer`.
"""
function batchdims(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return (ndims(layer) + 1):ndims(x)
end

"""
    batchsize(layer, x)

Batch sizes of `x`, with respect to `layer`.
"""
function batchsize(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return size(x)[batchdims(layer, x)]
end

"""
    batchmean(layer, x; wts = nothing)

Mean of `x` over batch dimensions, with weights `wts`.
"""
function batchmean(layer::AbstractLayer, x::AbstractArray; wts = nothing)
    @assert size(layer) == size(x)[1:ndims(layer)]
    μ = wmean(x; wts, dims = batchdims(layer, x))
    return reshape(μ, size(layer))
end
batchmean(::AbstractLayer, x::Number; wts::Nothing) = x

"""
    ∂energy(layer; stats...)

Derivative of average energy of configurations with respect to layer parameters,
where `stats...` refers to the sufficient statistics from samples required by the layer.
See [`sufficient_statistics`](@ref).
"""
function ∂energy end

"""
    ∂free_energy(layer, inputs = 0; wts = 1)

Unit activation moments, conjugate to layer parameters.
These are obtained by differentiating `free_energies` with respect to the layer parameters.
Averages over configurations (weigthed by `wts`).
"""
function ∂free_energy(layer::AbstractLayer, inputs::AbstractArray; wts = nothing)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    layer_eff = effective(layer, inputs)
    ∂Feff = ∂free_energy(layer_eff)
    if ndims(layer) == ndims(inputs)
        @assert isnothing(wts)
        @assert size(layer_eff) == size(layer)
        return ∂Feff
    else
        return map(∂Feff) do ∂fs
            @assert size(∂fs) == size(layer_eff)
            ∂ω = batchmean(layer, ∂fs; wts)
            @assert size(∂ω) == size(layer)
            ∂ω
        end
    end
end

function ∂free_energy(layer::AbstractLayer, input::Real; wts::Nothing = nothing)
    inputs = FillArrays.Fill(input, size(layer))
    return ∂free_energy(layer, inputs; wts)
end

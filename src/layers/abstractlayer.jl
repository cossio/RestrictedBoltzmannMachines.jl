abstract type AbstractLayer{N} end

Base.ndims(::AbstractLayer{N}) where {N} = N
Base.size(layer::AbstractLayer, d::Int) = size(layer)[d]

"""
    flatten(layer, x)

Returns a vectorized version of `x`.
"""
function flatten(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    if ndims(layer) == ndims(x)
        return vec(x)
    else
        return reshape(x, length(layer), :)
    end
end

"""
    effective(layer, inputs; β = 1)

Returns an effective layer which behaves as the original with the given `inputs` and
temperature.
"""
function effective(layer::AbstractLayer, input::Real; β::Real = true)
    inputs = Fill(input, size(layer))
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
    inputs = Fill(input, size(layer))
    return free_energy(layer, inputs; β)::Number
end

"""
    transfer_sample(layer, inputs = 0; β = 1)

Samples layer configurations conditioned on inputs.
"""
function transfer_sample(layer::AbstractLayer, inputs::Union{Real,AbstractArray}; β::Real=1)
    layer_eff = effective(layer, inputs; β)
    return transfer_sample(layer_eff)
end

"""
    transfer_mode(layer, inputs = 0)

Mode of unit activations.
"""
function transfer_mode(layer::AbstractLayer, inputs::Union{Real, AbstractArray})
    layer_eff = effective(layer, inputs)
    return transfer_mode(layer_eff)
end

"""
    transfer_mean(layer, inputs = 0; β = 1)

Mean of unit activations.
"""
function transfer_mean(layer::AbstractLayer, inputs::Union{Real,AbstractArray}; β::Real=1)
    layer_eff = effective(layer, inputs; β)
    return transfer_mean(layer_eff)
end

"""
    transfer_var(layer, inputs = 0; β = 1)

Variance of unit activations.
"""
function transfer_var(layer::AbstractLayer, inputs::Union{Real,AbstractArray}; β::Real=1)
    layer_ = effective(layer, inputs; β)
    return transfer_var(layer_)
end

"""
    transfer_std(layer, inputs = 0; β = 1)

Standard deviation of unit activations.
"""
function transfer_std(layer::AbstractLayer, inputs::Union{Real, AbstractArray}; β::Real=1)
    layer_eff = effective(layer, inputs; β)
    return transfer_std(layer_eff)
end

"""
    transfer_mean_abs(layer, inputs = 0; β = 1)

Mean of absolute value of unit activations.
"""
function transfer_mean_abs(layer::AbstractLayer, inputs::Union{Real,AbstractArray}; β::Real=1)
    layer_eff = effective(layer, inputs; β)
    return transfer_mean_abs(layer_eff)
end

"""
    free_energies(layer, inputs = 0; β = 1)

Cumulant generating function of units in layer (not reduced over layer dimensions).
"""
function free_energies(layer::AbstractLayer, inputs::Union{Real,AbstractArray}; β::Real=1)
    layer_eff = effective(layer, inputs; β)
    return free_energies(layer_eff) / β
end

"""
    suffstats(layer, data; [wts])

Computes the sufficient statistics of `layer` extracted from `data`.
"""
function suffstats end

"""
    batchdims(layer, x)

Indices of batch dimensions in `x`, with respect to `layer`.
"""
function batchdims(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return (ndims(layer) + 1):ndims(x)
end

"""
    batch_size(layer, x)

Batch sizes of `x`, with respect to `layer`.
"""
function batch_size(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return size(x)[batchdims(layer, x)]
end

"""
    batchmean(layer, x; wts = nothing)

Mean of `x` over batch dimensions, weigthed by `wts`.
"""
function batchmean(layer::AbstractLayer, x::AbstractArray; wts = nothing)
    @assert size(layer) == size(x)[1:ndims(layer)]
    if ndims(layer) == ndims(x)
        wts::Nothing
        return x
    else
        μ = wmean(x; wts, dims = batchdims(layer, x))
        return reshape(μ, size(layer))
    end
end

"""
    batchvar(layer, x; wts = nothing, [mean])

Variance of `x` over batch dimensions, weigthed by `wts`.
"""
function batchvar(
    layer::AbstractLayer, x::AbstractArray; wts = nothing, mean = batchmean(layer, x; wts)
)
    @assert size(layer) == size(x)[1:ndims(layer)] == size(mean)
    return batchmean(layer, (x .- mean).^2; wts)
end

"""
    batchstd(layer, x; wts = nothing, [mean])

Standard deviation of `x` over batch dimensions, weigthed by `wts`.
"""
function batchstd(
    layer::AbstractLayer, x::AbstractArray; wts = nothing, mean = batchmean(layer, x; wts)
)
    return sqrt.(batchvar(layer, x; wts, mean))
end

"""
    batchcov(layer, x; wts = nothing, [mean])

Covariance of `x` over batch dimensions, weigthed by `wts`.
"""
function batchcov(
    layer::AbstractLayer, x::AbstractArray; wts = nothing, mean = batchmean(layer, x; wts)
)
    @assert size(layer) == size(x)[1:ndims(layer)] == size(mean)
    ξ = flatten(layer, x .- mean)
    if isnothing(wts)
        w = I
    else
        @assert size(wts) == batch_size(layer, x)
        w = Diagonal(vec(wts))
    end
    C = ξ * w * ξ' / size(ξ, 2)
    return reshape(C, size(layer)..., size(layer)...)
end

"""
    ∂energy(layer, data; wts = nothing)
    ∂energy(layer, stats)

Derivative of average energy of `data` with respect to `layer` parameters.
In the second form, `stats` are the sufficient statistics of the layer.
See [`suffstats`](@ref).
"""
function ∂energy(layer::AbstractLayer, data::AbstractArray; wts=nothing)
    stats = suffstats(layer, data; wts)
    return ∂energy(layer, stats)
end

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
        wts::Nothing
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
    inputs = Fill(input, size(layer))
    return ∂free_energy(layer, inputs; wts)
end

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
        return reshape(x, length(layer), prod(size(x, d) for d in (ndims(layer) + 1):ndims(x)))
    end
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
    free_energy(layer, inputs = 0)

Cumulant generating function of layer, reduced over layer dimensions.
"""
function free_energy(layer::AbstractLayer, inputs::Union{Real,AbstractArray} = 0)
    F = free_energies(layer, inputs)
    if inputs isa Real || ndims(layer) == ndims(inputs)
        return sum(F)
    else
        f = sum(F; dims = 1:ndims(layer))
        return reshape(f, size(inputs)[(ndims(layer) + 1):end])
    end
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
        C = ξ * ξ' / size(ξ, 2)
    else
        @assert size(wts) == batch_size(layer, x)
        w = Diagonal(vec(wts))
        C = ξ * w * ξ' / sum(w)
    end
    return reshape(C, size(layer)..., size(layer)...)
end

"""
    total_mean_from_inputs(layer, inputs; wts = nothing)

Total mean of unit activations from inputs.
"""
function total_mean_from_inputs(
    layer::AbstractLayer, inputs::Union{Real,AbstractArray}; wts = nothing
)
    h_ave = mean_from_inputs(layer, inputs)
    return batchmean(layer, h_ave; wts)
end

"""
    total_var_from_inputs(layer, inputs; wts = nothing)

Total variance of unit activations from inputs.
"""
function total_var_from_inputs(
    layer::AbstractLayer, inputs::Union{Real,AbstractArray}; wts = nothing
)
    h_ave, h_var = meanvar_from_inputs(layer, inputs)
    ν_int = batchmean(layer, h_var; wts) # intrinsic noise
    ν_ext = batchvar(layer, h_ave; wts) # extrinsic noise
    return ν_int + ν_ext # law of total variance
end

"""
    total_meanvar_from_inputs(layer, inputs; wts = nothing)

Total mean and total variance of unit activations from inputs.
"""
function total_meanvar_from_inputs(
    layer::AbstractLayer, inputs::Union{Real,AbstractArray}; wts = nothing
)
    h_ave, h_var = meanvar_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave; wts)
    ν_int = batchmean(layer, h_var; wts) # intrinsic noise
    ν_ext = batchvar(layer, h_ave; wts, mean = μ) # extrinsic noise
    ν = ν_int + ν_ext # law of total variance
    return (μ = μ, ν = ν)
end

"""
    ∂energy(layer, data; wts = nothing)
    ∂energy(layer, stats)

Derivative of average energy of `data` with respect to `layer` parameters.
In the second form, `stats` are the pre-computed sufficient statistics
of the layer. See [`suffstats`](@ref).
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
function ∂free_energy(layer::AbstractLayer, inputs::Union{Real,AbstractArray} = 0; wts = nothing)
    ∂F = ∂free_energies(layer, inputs)
    return map(∂F) do ∂f
        batchmean(layer, ∂f; wts)
    end
end

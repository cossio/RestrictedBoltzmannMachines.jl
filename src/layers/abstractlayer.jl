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

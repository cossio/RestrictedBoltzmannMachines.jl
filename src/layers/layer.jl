export AbstractLayer
export checkdims, batchdims, batchindices, batchsize,
    energy, random, cgf, effective, fields, fieldtype,
    transfer_mode, transfer_mean, transfer_std, transfer_var, transfer_mean_abs

abstract type AbstractLayer{T,N} end
Base.ndims(::AbstractLayer{T,N}) where {T,N} = N
fieldtype(::AbstractLayer{T,N}) where {T,N} = T

"""
    checkdims(layer, x)

Checks dimensional consistency between `layer` and configuration `x`.
"""
function checkdims(layer::AbstractLayer, x::AbstractArray)
    ndims(x) ≥ ndims(layer) || dimserror()
    x_size = ntuple(d -> size(x, d), Val(ndims(layer)))
    x_size == size(layer) || dimserror()
end

function Base.size(layer::AbstractLayer, dim::Int)
    allequal(size.(fields(layer))...) ||
        throw(DimensionMismatch("Inconsistent parameter sizes"))
    return size(first(fields(layer)), dim)
end

function Base.size(layer::AbstractLayer)
    allequal(size.(fields(layer))...) ||
        throw(DimensionMismatch("Inconsistent parameter sizes"))
    return size(first(fields(layer)))
end

function Base.length(layer::AbstractLayer)
    allequal(size.(fields(layer))...) ||
        throw(DimensionMismatch("Inconsistent parameter sizes"))
    return length(first(fields(layer)))
end

dimserror() = throw(DimensionMismatch("Inconsistent dimensions"))
pardimserror() = throw(DimensionMismatch("Inconsistent parameter sizes"))

"""
    batchsize(layer, x)

Returns the batch sizes of the configurations in `x`, as implied by
the dimensions of `layer`.
"""
@generated batchsize(layer::AbstractLayer{T,N}, x::AbstractArray) where {N,T} =
    Expr(:tuple, (:(size(x, $d)) for d in N + 1 : ndims(x))...)

"""
    batchdims(layer, x)

Returns a tuple of the batch dimensions, statically determined from the types
of `layer` and `x`.
"""
@generated batchdims(layer::AbstractLayer{T,N}, x::AbstractArray) where {T,N} =
    Tuple(N + 1 : ndims(x))

"""
    batchndims(layer, x)

Statically determine the number of batch dimensions in `x`.
"""
@generated batchndims(layer::AbstractLayer{T,N}, x::AbstractArray) where {T,N} =
    ndims(x) - N

"""
    sitedims(layer)

Returns a tuple of the site dimensions of layer, statically.
"""
@generated sitedims(layer::AbstractLayer{T,N}) where {T,N} = Tuple(1:N)
siteindices(layer::AbstractLayer) = CartesianIndices(first(fields(layer)))
sitesize(layer::AbstractLayer) = size(layer)

"""
    batchindices(layer, x)

Returns a `CartesianIndices` over the batch indices of `x`.
"""
function batchindices(layer::AbstractLayer, x::AbstractArray)
    checkdims(layer, x)
    bsize = batchsize(layer, x)
    CartesianIndices(OneTo.(bsize))
end

"""
    energy(layer, x)

Energy of `layer` in configuration `x`.
"""
function energy(layer::AbstractLayer{T,N}, x) where {T,N}
    checkdims(layer, x)
    E = _energy(layer, x)
    return sumdropfirst(E, Val(N)) ./ 1 # convert zero-dim array to number
end

"""
    cgf(layer, I = 0, β = 1)

Cumulative generating function of layer conditioned on
input from other layer.
"""
function cgf(layer::AbstractLayer{T,N}, I = 0, β = 1) where {T,N}
    eff = effective(layer, I, β)
    Γ = _cgf(eff)
    return sumdropfirst(Γ, Val(N)) ./ β
end

"""
    random(layer, I = 0, β = 1)

Sample a random layer configuration.
"""
function random(layer::AbstractLayer, I = 0, β = 1)
    eff = effective(layer, I, β)
    return _random(eff)
end

"""
    transfer_mode(unit, I = 0)

Most likely unit state conditional on input from other layer.
"""
function transfer_mode(layer::AbstractLayer, I = 0, β = 1)
    eff = effective(layer, I, β)
    return _transfer_mode(eff)
end

"""
    transfer_mean(layer, I = 0, β = 1)

Mean over the configurations of `layer`, <h | v>
"""
function transfer_mean(layer::AbstractLayer, I = 0, β = 1)
    eff = effective(layer, I, β)
    return _transfer_mean(eff)
end

"""
    transfer_mean_abs(layer, I = 0, β = 1)

Mean over the absolute values of the configurations of `layer`, <|h| | v>.
"""
function transfer_mean_abs(layer::AbstractLayer, I = 0, β = 1)
    eff = effective(layer, I, β)
    return _transfer_mean_abs(eff)
end

"""
    transfer_std(layer, I = 0, β = 1)

Standard deviation over the configurations of `layer`.
"""
function transfer_std(layer::AbstractLayer, I = 0, β = 1)
    eff = effective(layer, I, β)
    return _transfer_std(eff)
end

"""
    transfer_var(layer, I = 0, β = 1)

Variance over the configurations of `layer`.
"""
function transfer_var(layer::AbstractLayer, I = 0, β = 1)
    eff = effective(layer, I, β)
    return _transfer_var(eff)
end

"""
    effective(layer, I = 0, β = 1)

Effective unit given input from other layer and inverse temperature β.
"""
function effective(layer::AbstractLayer, I = 0, β = 1)
    eff1 = effective_I(layer, I)
    eff2 = effective_β(eff1, β)
    return eff2
end

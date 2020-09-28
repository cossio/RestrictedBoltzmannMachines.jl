export AbstractLayer
export checkdims, batchdims, batchindices, batchsize,
    energy, random, cgf, effective, fields, fieldtype,
    transfer_mode, transfer_mean, transfer_std, transfer_var, transfer_mean_abs,
    transfer_pdf, transfer_cdf, transfer_logpdf, transfer_logcdf, transfer_entropy

abstract type AbstractLayer{T,N} end
Base.ndims(::AbstractLayer{T,N}) where {T,N} = N
Base.ndims(::Type{<:AbstractLayer{T,N}}) where {T,N} = N
fieldtype(::AbstractLayer{T,N}) where {T,N} = T
fieldtype(::Type{<:AbstractLayer{T,N}}) where {T,N} = T

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
    size(first(fields(layer)), dim)
end

function Base.size(layer::AbstractLayer)
    allequal(size.(fields(layer))...) ||
        throw(DimensionMismatch("Inconsistent parameter sizes"))
    size(first(fields(layer)))
end

function Base.length(layer::AbstractLayer)
    allequal(size.(fields(layer))...) ||
        throw(DimensionMismatch("Inconsistent parameter sizes"))
    length(first(fields(layer)))
end

dimserror() = throw(DimensionMismatch("Inconsistent dimensions"))
pardimserror() = throw(DimensionMismatch("Inconsistent parameter sizes"))

"""
    batchsize(layer, x)

Returns the batch sizes of the configurations in `x`, as implied by
the dimensions of `layer`.
"""
@generated batchsize(::AbstractLayer{T,N}, x::AbstractArray) where {N,T} =
    Expr(:tuple, (:(size(x, $d)) for d in N + 1 : ndims(x))...)

"""
    batchdims(layer, x)

Returns a tuple of the batch dimensions, statically determined from the types
of `layer` and `x`.
"""
@generated batchdims(layer::AbstractLayer, x::AbstractArray) = Tuple(ndims(layer) + 1 : ndims(x))

"""
    batchndims(layer, x)

Statically determine the number of batch dimensions in `x`.
"""
@generated batchndims(layer::AbstractLayer, x::AbstractArray) = ndims(x) - ndims(layer)

"""
    sitedims(layer)

Returns a tuple of the site dimensions of layer, statically.
"""
@generated sitedims(layer::AbstractLayer) = Tuple(1:ndims(layer))
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

#= We have three energy functions: energy, _energy, and __energy.
__energy returns an array of energies for each unit in each batch.
_energy reduces across the layer and returns an array of layer energies for each batch.
energy also takes care of converting zero-dimensional arrays to scalars
(for the single-batch case). =#

"""
    energy(layer, x)

Energy of `layer` in configuration `x`.
"""
energy(layer::AbstractLayer, x::AbstractArray) = scalarize(_energy(layer, x))
function _energy(layer::AbstractLayer, x::AbstractArray)
    checkdims(layer, x)
    sumdropfirst(__energy(layer, x), Val(ndims(layer)))
end

"""
    cgf(layer, I = 0, β = 1)

Cumulative generating function of layer conditioned on
input from other layer.
"""
cgf(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    scalarize(_cgf(layer, I, β))
function _cgf(layer::AbstractLayer, I = 0, β = 1)
    Γ = __cgf(effective(layer, I, β)) ./ β
    return sumdropfirst(Γ, Val(ndims(layer)))
end

"""
    random(layer, I = 0, β = 1)

Sample a random layer configuration.
"""
random(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    _random(effective(layer, I, β))

"""
    transfer_mode(unit, I = 0)

Most likely unit state conditional on input from other layer.
"""
transfer_mode(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    _transfer_mode(effective(layer, I, β))

"""
    transfer_mean(layer, I = 0, β = 1)

Mean over the configurations of `layer`, <h | v>
"""
transfer_mean(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    _transfer_mean(effective(layer, I, β))

"""
    transfer_mean_abs(layer, I = 0, β = 1)

Mean over the absolute values of the configurations of `layer`, <|h| | v>.
"""
transfer_mean_abs(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    _transfer_mean_abs(effective(layer, I, β))

"""
    transfer_std(layer, I = 0, β = 1)

Standard deviation over the configurations of `layer`.
"""
transfer_std(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    _transfer_std(effective(layer, I, β))

"""
    transfer_var(layer, I = 0, β = 1)

Variance over the configurations of `layer`.
"""
transfer_var(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    _transfer_var(effective(layer, I, β))

transfer_pdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(transfer_logpdf(layer, x, I, β))
transfer_cdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(transfer_logcdf(effective(layer, I, β), x))
transfer_survival(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(transfer_logsurvival(layer, x, I, β))

transfer_logpdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    scalarize(_transfer_logpdf(layer, x, I, β))
transfer_logcdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    scalarize(_transfer_logcdf(layer, x, I, β))
transfer_logsurvival(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    scalarize(_transfer_logsurvival(layer, x, I, β))
transfer_entropy(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    scalarize(_transfer_entropy(layer, I, β))

_transfer_pdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(_transfer_logpdf(layer, x, I, β))
_transfer_cdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(_transfer_logcdf(layer, x, I, β))
_transfer_survival(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(_transfer_logsurvival(layer, x, I, β))

_transfer_logpdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    sumdropfirst(__transfer_logpdf(layer, x, I, β), Val(ndims(layer)))
_transfer_logcdf(layer::AbstractLayer, x::AbstractArray, I = 0, β = 1) =
    sumdropfirst(__transfer_logcdf(effective(layer, I, β), x), Val(ndims(layer)))
_transfer_logsurvival(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    sumdropfirst(__transfer_logsurvival(layer, x, I, β), Val(ndims(layer)))
_transfer_entropy(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    sumdropfirst(__transfer_entropy(layer, I, β), Val(ndims(layer)))

__transfer_logpdf(layer::AbstractLayer, x::AbstractArray, I::Numeric, β::Numeric = 1) =
    __transfer_logpdf(effective(layer, I, β), x)
__transfer_logcdf(layer::AbstractLayer, x::AbstractArray, I::Numeric, β::Numeric = 1) =
    __transfer_logcdf(effective(layer, I, β), x)
__transfer_logsurvival(layer::AbstractLayer, x::AbstractArray, I::Numeric, β::Numeric = 1) =
    __transfer_logsurvival(effective(layer, I, β), x)

__transfer_pdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(__transfer_logpdf(layer, x, I, β))
__transfer_cdf(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(__transfer_logcdf(layer, x, I, β))
__transfer_survival(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    exp.(__transfer_logsurvival(layer, x, I, β))

__transfer_entropy(layer::AbstractLayer, I::Numeric, β::Numeric = 1) =
    __transfer_entropy(effective(layer, I, β))

transfer_mills(layer::AbstractLayer, x::AbstractArray, I::Numeric = 0, β::Numeric = 1) =
    _transfer_mills(effective(layer, I, β), x)

"""
    effective(layer, I = 0, β = 1)

Effective unit given input from other layer and inverse temperature β.
"""
effective(layer::AbstractLayer, I::Numeric = 0, β::Numeric = 1) =
    effective_β(effective_I(layer, I), β)

export AbstractLayer, AbstractHomogeneousLayer
export AbstractDiscreteLayer, AbstractContinuousLayer
export checkdims, batchdims, batchindices, batchsize, fieldtype
export energy, random, cgf, effective, fields
export transfer_mode, transfer_mean, transfer_std, transfer_var, transfer_mean_abs
export transfer_pdf, transfer_cdf, transfer_logpdf, transfer_logcdf, transfer_entropy

abstract type AbstractLayer end
abstract type AbstractHomogeneousLayer{T,N} <: AbstractLayer end
abstract type AbstractDiscreteLayer{T,N} <: AbstractHomogeneousLayer{T,N} end
abstract type AbstractContinuousLayer{T,N} <: AbstractHomogeneousLayer{T,N} end
Base.ndims(::AbstractHomogeneousLayer{T,N}) where {T,N} = N
Base.ndims(::Type{<:AbstractHomogeneousLayer{T,N}}) where {T,N} = N
fieldtype(::AbstractHomogeneousLayer{T,N}) where {T,N} = T
fieldtype(::Type{<:AbstractHomogeneousLayer{T,N}}) where {T,N} = T

"""
    checkdims(layer, x)

Checks dimensional consistency between `layer` and configuration `x`.
"""
function checkdims(layer::AbstractHomogeneousLayer, x::NumArray)
    ndims(x) ≥ ndims(layer) || dimserror()
    x_size = ntuple(d -> size(x, d), Val(ndims(layer)))
    x_size == size(layer) || dimserror()
end

function Base.size(layer::AbstractHomogeneousLayer, dim::Int)
    allequal(size.(fields(layer))...) ||
        throw(DimensionMismatch("Inconsistent parameter sizes"))
    size(first(fields(layer)), dim)
end

function Base.size(layer::AbstractHomogeneousLayer)
    allequal(size.(fields(layer))...) ||
        throw(DimensionMismatch("Inconsistent parameter sizes"))
    size(first(fields(layer)))
end

function Base.length(layer::AbstractHomogeneousLayer)
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
@generated batchsize(::AbstractHomogeneousLayer{T,N}, x::NumArray) where {N,T} =
    Expr(:tuple, (:(size(x, $d)) for d in N + 1 : ndims(x))...)

"""
    batchdims(layer, x)

Returns a tuple of the batch dimensions, statically determined from the types
of `layer` and `x`.
"""
@generated batchdims(layer::AbstractHomogeneousLayer, x::NumArray) = Tuple(ndims(layer) + 1 : ndims(x))

"""
    batchndims(layer, x)

Statically determine the number of batch dimensions in `x`.
"""
@generated batchndims(layer::AbstractHomogeneousLayer, x::NumArray) = ndims(x) - ndims(layer)

"""
    sitedims(layer)

Returns a tuple of the site dimensions of layer, statically.
"""
@generated sitedims(layer::AbstractHomogeneousLayer) = Tuple(1:ndims(layer))
siteindices(layer::AbstractHomogeneousLayer) = CartesianIndices(first(fields(layer)))
sitesize(layer::AbstractHomogeneousLayer) = size(layer)

"""
    batchindices(layer, x)

Returns a `CartesianIndices` over the batch indices of `x`.
"""
function batchindices(layer::AbstractHomogeneousLayer, x::NumArray)
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
energy(layer::AbstractHomogeneousLayer, x::NumArray) = scalarize(_energy(layer, x))
function _energy(layer::AbstractHomogeneousLayer, x::NumArray)
    checkdims(layer, x)
    sumdropfirst(__energy(layer, x), Val(ndims(layer)))
end

"""
    cgf(layer, I = 0, β = 1)

Cumulative generating function of layer conditioned on
input from other layer.
"""
cgf(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) = scalarize(_cgf(layer, I, β))
function _cgf(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1)
    Γ = __cgf(effective(layer, I, β)) ./ β
    return sumdropfirst(Γ, Val(ndims(layer)))
end

"""
    random(layer, I = 0, β = 1)

Sample a random layer configuration.
"""
random(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) = _random(effective(layer, I, β))

"""
    transfer_mode(unit, I = 0)

Most likely unit state conditional on input from other layer.
"""
transfer_mode(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) = _transfer_mode(effective(layer, I, β))

"""
    transfer_mean(layer, I = 0, β = 1)

Mean over the configurations of `layer`, <h | v>
"""
transfer_mean(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) =
    _transfer_mean(effective(layer, I, β))

"""
    transfer_mean_abs(layer, I = 0, β = 1)

Mean over the absolute values of the configurations of `layer`, <|h| | v>.
"""
transfer_mean_abs(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) =
    _transfer_mean_abs(effective(layer, I, β))

"""
    transfer_std(layer, I = 0, β = 1)

Standard deviation over the configurations of `layer`.
"""
transfer_std(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) =
    _transfer_std(effective(layer, I, β))

"""
    transfer_var(layer, I = 0, β = 1)

Variance over the configurations of `layer`.
"""
transfer_var(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) =
    _transfer_var(effective(layer, I, β))

transfer_pdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(transfer_logpdf(layer, x, I, β))
transfer_cdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(transfer_logcdf(effective(layer, I, β), x))
transfer_survival(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(transfer_logsurvival(layer, x, I, β))

transfer_logpdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    scalarize(_transfer_logpdf(layer, x, I, β))
transfer_logcdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    scalarize(_transfer_logcdf(layer, x, I, β))
transfer_logsurvival(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    scalarize(_transfer_logsurvival(layer, x, I, β))
transfer_entropy(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) =
    scalarize(_transfer_entropy(layer, I, β))

_transfer_pdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(_transfer_logpdf(layer, x, I, β))
_transfer_cdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(_transfer_logcdf(layer, x, I, β))
_transfer_survival(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(_transfer_logsurvival(layer, x, I, β))

_transfer_logpdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    sumdropfirst(__transfer_logpdf(layer, x, I, β), Val(ndims(layer)))
_transfer_logcdf(layer::AbstractHomogeneousLayer, x::NumArray, I = 0, β = 1) =
    sumdropfirst(__transfer_logcdf(effective(layer, I, β), x), Val(ndims(layer)))
_transfer_logsurvival(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    sumdropfirst(__transfer_logsurvival(layer, x, I, β), Val(ndims(layer)))
_transfer_entropy(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) =
    sumdropfirst(__transfer_entropy(layer, I, β), Val(ndims(layer)))

__transfer_logpdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num, β::Num = 1) =
    __transfer_logpdf(effective(layer, I, β), x)
__transfer_logcdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num, β::Num = 1) =
    __transfer_logcdf(effective(layer, I, β), x)
__transfer_logsurvival(layer::AbstractHomogeneousLayer, x::NumArray, I::Num, β::Num = 1) =
    __transfer_logsurvival(effective(layer, I, β), x)

__transfer_pdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(__transfer_logpdf(layer, x, I, β))
__transfer_cdf(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(__transfer_logcdf(layer, x, I, β))
__transfer_survival(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    exp.(__transfer_logsurvival(layer, x, I, β))

__transfer_entropy(layer::AbstractHomogeneousLayer, I::Num, β::Num = 1) =
    __transfer_entropy(effective(layer, I, β))

transfer_mills(layer::AbstractHomogeneousLayer, x::NumArray, I::Num = 0, β::Num = 1) =
    _transfer_mills(effective(layer, I, β), x)

"""
    effective(layer, I = 0, β = 1)

Effective unit given input from other layer and inverse temperature β.
"""
effective(layer::AbstractHomogeneousLayer, I::Num = 0, β::Num = 1) =
    effective_β(effective_I(layer, I), β)

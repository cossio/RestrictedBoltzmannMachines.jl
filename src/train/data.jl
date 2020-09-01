# Based on https://github.com/FluxML/Flux.jl/pull/1221/
# but implementing a dataset that loops infinitely instead of by epochs.

struct Data{D<:NamedTuple}
    tensors::D
    batchsize::Int
    nobs::Int
end

function Data(tensors::NamedTuple; batchsize::Int = 1)
    n = _nobs(tensors)
    @assert 0 ≤ batchsize ≤ n "batchsize=$batchsize invalid with nobs=$n"
    return Data(tensors, batchsize, n)
end

Data() =  Data((;); batchsize = 0)

Base.@propagate_inbounds function Base.iterate(d::Data, (i, shuffled) = (0, _shuffleobs(d)))
    if i ≥ d.nobs - d.batchsize + 1
        i = 0
        shuffled = _shuffleobs(d)
    end
    batch = _getobs(shuffled, i .+ (1:d.batchsize))
    return (batch, (i + d.batchsize, shuffled))
end

function _nobs(tensors::NamedTuple)
    isempty(tensors) && return 0
    @assert length(tensors) > 0
    ns = values(map(_nobs, tail(tensors)))
    all(ns .== first(ns)) || throw(DimensionMismatch("All data tensors must contain the same number of observations"))
    return first(ns)
end
_nobs(tensor::AbstractArray) = last(size(tensor))
_getobs(tensor::AbstractArray, i) = tensor[ntuple(i -> Colon(), Val(ndims(tensor) - 1))..., i]
_getobs(tensors::NamedTuple, i) = map(Base.Fix2(_getobs, i), tensors)
_shuffleobs(d::Data) = _getobs(d.tensors, randperm(d.nobs))
Base.eltype(::Data{D}) where {D} = D
_seldim(tensor::AbstractArray, ::Val{d}, i) where {d} = selectdim(tensor, d, i)

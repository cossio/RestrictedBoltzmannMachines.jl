# Based on https://github.com/FluxML/Flux.jl/pull/1221/
# but implementing a dataset that loops infinitely instead of by epochs.

function _nobs(ds::AbstractArray...)
    sz = map(d -> size(d)[end], ds)
    @assert all(sz .== first(sz))
    return first(sz)
end

_getobs(i, ds::AbstractArray...) = map(d -> collect(selectdim(d, ndims(d), i)), ds)

"""
    minibatches(datas...; batchsize)

Splits the given `datas` into minibatches. Each minibatch is a tuple where
each entry is a minibatch from the corresponding `data` within `datas`.
All minibatches are of the same size `batchsize` (if necessary repeating
some samples at the last minibatches).
"""
function minibatches(ds::AbstractArray...; batchsize::Int)
    nobs = _nobs(ds...)
    perm = randperm(nobs)
    slices = minibatches(nobs; batchsize = batchsize)
    batches = [_getobs(perm[b], ds...) for b in slices]
    return batches
end

"""
    minibatch_count(data; batchsize)

Number of minibatches.
"""
function minibatch_count(ds::AbstractArray...; batchsize::Int)
    return minibatch_count(_nobs(ds...); batchsize = batchsize)
end

"""
    minibatches(nobs; batchsize)

Partition `nobs` into minibatches of length `n`.
If necesarry repeats some observations to complete last batches.
(Therefore all batches are of the same size `n`).
"""
function minibatches(nobs::Int; batchsize::Int)
    @assert nobs > 0 && batchsize > 0
    nbatches = minibatch_count(nobs; batchsize = batchsize)
    idx = mod1.(1:(nbatches * batchsize), nobs)
    return [idx[b:(b + batchsize - 1)] for b in 1:batchsize:length(idx)]
end

"""
    minibatch_count(nobs; batchsize)

Number of minibatches.
"""
function minibatch_count(nobs::Int; batchsize::Int)
    @assert nobs > 0 && batchsize > 0
    return cld(nobs, batchsize)
end

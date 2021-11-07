# Based on https://github.com/FluxML/Flux.jl/pull/1221/
# but implementing a dataset that loops infinitely instead of by epochs.

function _nobs(ds::AbstractArray...)
    sz = map(d -> size(d, ndims(d)), ds)
    @assert all(sz .== first(sz))
    return first(sz)
end

_getobs(i, ds::AbstractArray...) = map(d -> collect(selectdim(d, ndims(d), i)), ds)

function minibatches(ds::AbstractArray...; batchsize::Int, full = false)
    nobs = _nobs(ds...)
    perm = randperm(nobs)
    slices = minibatches(nobs; batchsize = batchsize, full = full)
    batches = [_getobs(perm[b], ds...) for b in slices]
    return batches
end

"""
    minibatch_count(data; batchsize, full = false)

Number of minibatches.
"""
function minibatch_count(ds::AbstractArray...; batchsize::Int, full::Bool = false)
    return minibatch_count(_nobs(ds...); batchsize = batchsize, full = full)
end

"""
    minibatches(nobs; batchsize, full = false)

Partition `samples` into minibatches of length `n`.
If `full` is `true`, all slices are of length `n`, otherwise
the last slice can be smaller.
"""
function minibatches(nobs::Int; batchsize::Int, full::Bool = false)
    @assert nobs ≥ 0 && batchsize ≥ 0
    if full
        endidx = fld(nobs, batchsize) * batchsize
    else
        endidx = min(cld(nobs, batchsize), nobs) * batchsize
    end
    return [b:min(b + batchsize - 1, nobs) for b = 1:batchsize:endidx]
end

"""
    minibatch_count(nobs; batchsize, full = false)

Number of minibatches.
"""
function minibatch_count(nobs::Int; batchsize::Int, full::Bool = false)
    if full
        endidx = fld(nobs, batchsize) * batchsize
    else
        endidx = min(cld(nobs, batchsize), nobs) * batchsize
    end
    return length(1:batchsize:endidx)
end

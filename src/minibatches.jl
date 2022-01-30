function _nobs(d::AbstractArray, ds::Union{AbstractArray, Nothing}...)
    Bs = map(_nobs, ds)
    @assert all((Bs .== _nobs(d)) .| isnothing.(ds))
    return _nobs(d)
end
_nobs(d::AbstractArray) = size(d, ndims(d))
_nobs(::Nothing...) = nothing
_selobs(i, d::AbstractArray) = collect(selectdim(d, ndims(d), i))
_selobs(i, ::Nothing) = nothing
_getobs(i, ds::Union{AbstractArray, Nothing}...) = map(d -> _selobs(i, d), ds)

"""
    minibatches(datas...; batchsize)

Splits the given `datas` into minibatches. Each minibatch is a tuple where
each entry is a minibatch from the corresponding `data` within `datas`.
All minibatches are of the same size `batchsize` (if necessary repeating
some samples at the last minibatches).
"""
function minibatches(
    ds::Union{AbstractArray, Nothing}...; batchsize::Int, shuffle::Bool = true
)
    nobs = _nobs(ds...)
    slices = minibatches(nobs; batchsize = batchsize, shuffle = shuffle)
    batches = [_getobs(idx, ds...) for idx in slices]
    return batches
end

"""
    minibatch_count(data; batchsize)

Number of minibatches.
"""
function minibatch_count(ds::Union{AbstractArray, Nothing}...; batchsize::Int)
    return minibatch_count(_nobs(ds...); batchsize = batchsize)
end

"""
    minibatches(nobs; batchsize, shuffle = true)

Partition `nobs` into minibatches of length `n`.
If necessary repeats some observations to complete last batches.
(Therefore all batches are of the same size `n`).
"""
function minibatches(nobs::Int; batchsize::Int, shuffle::Bool = true)
    @assert nobs > 0 && batchsize > 0
    nbatches = minibatch_count(nobs; batchsize = batchsize)
    idx = mod1.(1:(nbatches * batchsize), nobs)
    if shuffle
        Random.shuffle!(idx)
    end
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

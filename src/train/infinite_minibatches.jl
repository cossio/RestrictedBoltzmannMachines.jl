"""
    infinite_minibatches(ds...; batchsize, shuffle = true)

Infinite iterator over minibatches of `batchsize` observations from `ds`, which are
indexed jointly along their last dimension. With `shuffle = true` each minibatch is an
independent uniformly random subset of the observations, drawn without replacement within
the minibatch (so a `batchsize` equal to the number of observations yields the full data
every iteration). With `shuffle = false` the minibatches cycle through the data in order,
dropping the trailing partial batch.
"""
function infinite_minibatches(ds::AbstractArray...; batchsize::Int, shuffle::Bool = true)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    n = size(first(ds))[end]
    all(d -> size(d)[end] == n, ds) ||
        throw(DimensionMismatch("all arrays must have the same number of observations"))
    batchsize ≤ n ||
        throw(ArgumentError("batchsize ($batchsize) exceeds the number of observations ($n)"))
    return (_getobs(ds, _minibatch_indices(k, n, batchsize, shuffle)) for k in Iterators.countfrom(0))
end

_getobs(ds::Tuple, idx) = map(d -> d[.., idx], ds)

function _minibatch_indices(k::Int, n::Int, batchsize::Int, shuffle::Bool)
    shuffle && return sample(1:n, batchsize; replace = false)
    offset = (k % (n ÷ batchsize)) * batchsize
    return (offset + 1):(offset + batchsize)
end

"""
    validate_wts(wts)

Asserts that the data weights `wts` are finite and positive.
"""
validate_wts(::Ones{<:Real}) = nothing

function validate_wts(wts::AbstractArray{<:Real})
    @assert all(w -> isfinite(w) && w > 0, wts) "wts must contain only finite, positive values"
    return nothing
end

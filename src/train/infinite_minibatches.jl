function nobs(d::AbstractArray, ds::AbstractArray...)
    n = nobs(d)
    @assert all(d -> nobs(d) == n, ds)
    return n
end

nobs(d::AbstractArray) = size(d, ndims(d))

getobs(i, ds::AbstractArray...) = map(d -> d[.., i], ds)

shuffleobs(ds::AbstractArray...) = getobs(randperm(nobs(ds...)), ds...)

struct InfiniteMinibatchIterator{T}
    data::T
    batchsize::Int
    shuffle::Bool
end

function Base.iterate(iter::InfiniteMinibatchIterator)
    iter.batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    if iter.batchsize > nobs(iter.data...)
        return nothing
    else
        if iter.shuffle
            shuffled = shuffleobs(iter.data...)
        else
            shuffled = iter.data
        end
        return iterate(iter, (i = 1, shuffled))
    end
end

function Base.iterate(iter::InfiniteMinibatchIterator, (i, shuffled))
    if i + iter.batchsize - 1 > nobs(iter.data...)
        return iterate(iter) # restart iteration
    else
        items = getobs(i:(i + iter.batchsize - 1), shuffled...)
        return items, (i + iter.batchsize, shuffled)
    end
end

function infinite_minibatches(ds::AbstractArray...; batchsize::Int, shuffle::Bool = true)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    return InfiniteMinibatchIterator(ds, batchsize, shuffle)
end

# Lazy uniform sample weights when the user gives none.
_default_weights(wts::AbstractVector, ::AbstractArray) = wts
_default_weights(::Nothing, data::AbstractArray) = Ones{Bool}(size(data, ndims(data)))

#= Per-run weight hygiene: validate the weights and drop zero-weight samples,
so the per-iteration kernels can reduce with plain matmuls (no `NaN * 0` from
zero-weight samples). Weights are not rescaled: extreme finite weights (near
`floatmax`, or needing wider-than-Float64 accumulation) can overflow the
plain reductions, which is accepted. Lazy uniform `Ones` weights skip the
hygiene by dispatch. Returns the prepared `(data, wts)`, the mean of the
prepared weights (used to bias-correct minibatch gradients), and the
batchsize clamped to the number of remaining samples. =#
function _prepare_training_data(
        data::AbstractArray,
        wts::AbstractVector;
        batchsize::Int,
    )
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    wts = _validate_weights(wts)
    data, wts = _filter_zero_weights(data, wts)
    return data, wts, mean(wts), min(batchsize, length(wts))
end

# Real-valued lazy uniform weights are trivially valid; anything else
# (including complex-valued `Ones`) goes through the elementwise checks.
_validate_weights(wts::Ones{<:Real}) = wts

function _validate_weights(wts::AbstractArray)
    all(w -> w isa Real && isfinite(w) && w ≥ 0, wts) ||
        throw(ArgumentError("wts must contain only finite, nonnegative real values"))
    any(w -> !iszero(w), wts) ||
        throw(ArgumentError("wts must contain at least one positive weight"))
    return wts
end

_filter_zero_weights(data::AbstractArray, wts::Ones{<:Real}) = (data, wts)

function _filter_zero_weights(data::AbstractArray, wts::AbstractVector)
    positive = map(w -> !iszero(w), wts)
    npositive = count(positive)
    npositive == length(wts) && return data, wts

    # GPUArrays do not generally support logical indexing. Transfer only
    # the mask used to build indices; indexing preserves the data backend.
    positive_indices = findall(adapt(Array, positive))
    return getobs(positive_indices, data, wts)
end

# mean(wd) / mean(wts), the bias correction for a weighted minibatch: a
# scalar in the weights' own precision, applied in the gradient eltype at its
# use site.
_batch_weight(wd::AbstractVector, wts_mean::Real) = mean(wd) / wts_mean

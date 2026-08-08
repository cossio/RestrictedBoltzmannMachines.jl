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

#= Per-run weight checks. Weights are used as given: they are not rescaled
(extreme finite weights can overflow the plain reductions), and zero-weight
observations are not removed — dropping them beforehand is the caller's
responsibility. Returns the mean weight (used to bias-correct minibatch
gradients) and the batchsize clamped to the number of samples. =#
function _prepare_training_data(
        data::AbstractArray,
        wts::AbstractVector;
        batchsize::Int,
    )
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    _validate_weights(wts)
    return mean(wts), min(batchsize, length(wts))
end

# Nonempty real-valued lazy uniform weights are trivially valid; anything else
# (including complex-valued `Ones`) goes through the elementwise checks. Empty
# weights have no positive weight, whatever their type.
function _validate_weights(wts::Ones{<:Real})
    isempty(wts) &&
        throw(ArgumentError("wts must contain at least one positive weight"))
    return wts
end

function _validate_weights(wts::AbstractArray)
    all(w -> w isa Real && isfinite(w) && w ≥ 0, wts) ||
        throw(ArgumentError("wts must contain only finite, nonnegative real values"))
    any(w -> !iszero(w), wts) ||
        throw(ArgumentError("wts must contain at least one positive weight"))
    return wts
end

# mean(wd) / mean(wts), the bias correction for a weighted minibatch: a
# scalar in the weights' own precision, applied in the gradient eltype at its
# use site.
_batch_weight(wd::AbstractVector, wts_mean::Real) = mean(wd) / wts_mean

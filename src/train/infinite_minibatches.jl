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

#= Per-run weight hygiene: validate the weights, normalize by the largest
weight, and drop zero-weight samples, so the per-iteration kernels can reduce
with plain matmuls (no per-iteration overflow armor, no `NaN * 0` from
zero-weight samples). Filtering runs after normalization so relative weights
that underflow `float(eltype(wts))` are dropped like exact zeros, instead of
surviving into minibatches whose weights sum to zero. Lazy uniform `Ones`
weights skip all of it by dispatch. Returns the prepared `(data, wts)`, the
Float64 mean of the prepared weights (used to bias-correct minibatch
gradients), and the batchsize clamped to the number of remaining samples. =#
function _prepare_training_data(
        data::AbstractArray,
        wts::AbstractVector;
        batchsize::Int,
    )
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    wts = _normalize_weights(_validate_weights(wts))
    data, wts = _filter_zero_weights(data, wts)
    # Scalar Float64 accumulation; it never touches the array eltypes.
    wts_mean = sum(Float64, wts) / length(wts)
    return data, wts, wts_mean, min(batchsize, length(wts))
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

# Normalizing by the largest weight bounds the prepared weights by one, so
# extreme finite weights cannot overflow the training-loop reductions. The
# division stays in `float(eltype(wts))` to avoid promoting the arrays the
# weights later multiply; uniform `Ones` weights stay lazy.
_normalize_weights(wts::Ones{<:Real}) = wts
_normalize_weights(wts::AbstractArray) = wts ./ maximum(wts)

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

# mean(wd) / mean(wts), the bias correction for a weighted minibatch, as a
# Float64 scalar (converted to the gradient eltype at its use site).
_batch_weight(wd::AbstractVector, wts_mean::Float64) = sum(Float64, wd) / (length(wd) * wts_mean)

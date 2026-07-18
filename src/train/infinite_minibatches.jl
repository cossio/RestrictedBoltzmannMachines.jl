function nobs(d::AbstractArray, ds::Union{AbstractArray, Nothing}...)
    n = nobs(d)
    ns = nobs(ds...)
    @assert n == ns || isnothing(ns)
    return n
end

nobs(::Nothing, ds::Union{AbstractArray, Nothing}...) = nobs(ds...)
nobs(d::AbstractArray) = size(d, ndims(d))
nobs(::Nothing) = nothing
nobs() = nothing

getobs(i, ds::Union{AbstractArray, Nothing}...) = map(ds) do d
    isnothing(d) ? nothing : d[.., i]
end

function shuffleobs(ds::Union{AbstractArray,Nothing}...)
    n = nobs(ds...)
    if isnothing(n)
        return ds
    else
        i = randperm(n)
        return getobs(i, ds...)
    end
end

struct InfiniteMinibatchIterator{T}
    data::T
    batchsize::Int
    shuffle::Bool
end

function Base.iterate(iter::InfiniteMinibatchIterator)
    iter.batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    n = nobs(iter.data...)
    if isnothing(n) || iter.batchsize > n
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

function infinite_minibatches(
    ds::Union{AbstractArray, Nothing}...; batchsize::Int, shuffle::Bool = true
)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    return InfiniteMinibatchIterator(ds, batchsize, shuffle)
end

function _prepare_training_data(
    data::AbstractArray,
    wts::Union{AbstractVector, Nothing};
    batchsize::Int,
)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    isnothing(wts) && return data, wts, nothing, batchsize

    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    all(w -> w isa Real && isfinite(w) && w ≥ 0, wts) ||
        throw(ArgumentError("wts must contain only finite, nonnegative real values"))

    positive = map(w -> !iszero(w), wts)
    npositive = count(positive)
    npositive > 0 ||
        throw(ArgumentError("wts must contain at least one positive weight"))

    if npositive < length(wts)
        # GPUArrays do not generally support logical indexing. Transfer only
        # the mask used to build indices; indexing preserves the data backend.
        positive_indices = findall(adapt(Array, positive))
        data, wts = getobs(positive_indices, data, wts)
    end

    # Cache the overall weight scale and mean, so minibatch gradients can be
    # bias-corrected without overflowing on extreme finite weights.
    scale = 1.0 * float(maximum(wts))
    normalization = (; scale, mean = mean(wts ./ scale))

    return data, wts, normalization, min(batchsize, npositive)
end

_batch_weight(::Nothing, ::Nothing) = 1

# mean(wd) / mean(wts), overflow-safe: batch weights are a subset of the
# training weights, so the global scale bounds them and its wide type
# propagates through the broadcast.
function _batch_weight(wd::AbstractVector, normalization::NamedTuple)
    return mean(wd ./ normalization.scale) / normalization.mean
end

struct _DefaultMoments end
const _DEFAULT_MOMENTS = _DefaultMoments()

struct _DefaultFantasy end
const _DEFAULT_FANTASY = _DefaultFantasy()

struct _DefaultOptimizerState end
const _DEFAULT_OPTIMIZER_STATE = _DefaultOptimizerState()

struct _DefaultChainCount end
const _DEFAULT_CHAIN_COUNT = _DefaultChainCount()

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

function _training_weight_accumulator_type(::Type{T}) where {T}
    F = float(T)
    # One wider IEEE type can represent the ratio between any two finite,
    # positive Float16 or Float32 values without underflow.
    return F === Float16 ? Float32 : F === Float32 ? Float64 : F
end

function _prepare_training_data(
    data::AbstractArray,
    wts::Union{AbstractVector, Nothing};
    batchsize::Int,
)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    isnothing(wts) && return data, wts, wts, nothing, batchsize

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

    # Weighted training is invariant to a common scale factor. Normalize before
    # moments and gradients so finite weights cannot overflow their sum/mean.
    # Keep the raw weights separately so callbacks continue to receive them.
    scale = maximum(wts)
    T = _training_weight_accumulator_type(typeof(scale))
    training_wts = T.(wts) ./ T(scale)
    normalization = (; scale, mean = mean(training_wts))

    return data, wts, training_wts, normalization, min(batchsize, npositive)
end

function _prepare_training_batch(
    wts::Nothing,
    normalization::Nothing,
)
    return wts, 1
end

function _prepare_training_batch(
    wts::AbstractVector,
    normalization::NamedTuple,
)
    scale = maximum(wts)
    T = promote_type(
        _training_weight_accumulator_type(typeof(scale)),
        typeof(normalization.mean),
    )
    training_wts = T.(wts) ./ T(scale)
    batch_weight = (T(scale) / T(normalization.scale)) *
        (mean(training_wts) / normalization.mean)
    return training_wts, batch_weight
end

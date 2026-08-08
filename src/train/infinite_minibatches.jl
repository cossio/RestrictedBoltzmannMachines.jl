#= Minibatch iteration is delegated to MLUtils.DataLoader. `Iterators.cycle`
restarts the loader whenever an epoch is exhausted, and the loader reshuffles on
each restart, so the stream is infinite with a fresh permutation per epoch.
`partial = false` drops the trailing incomplete batch, so every minibatch holds
exactly `batchsize` observations; in particular the stream is empty when
`batchsize` exceeds the number of observations (DataLoader would instead clamp
the batch size with a warning, so that case is guarded before the loader). =#

function infinite_minibatches(
        data::AbstractArray, wts::AbstractVector; batchsize::Int, shuffle::Bool = true
    )
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    batchsize ≤ numobs((data, wts)) || return ()
    return Iterators.cycle(DataLoader((data, wts); batchsize, shuffle, partial = false))
end

# MLUtils defines no observation semantics for `nothing` (and extending
# `numobs`/`getobs` to `Nothing` would be type piracy), so the unweighted case
# gets its own loader, normalized to yield `(vd, nothing)` like the weighted one.
function infinite_minibatches(
        data::AbstractArray, wts::Nothing; batchsize::Int, shuffle::Bool = true
    )
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    batchsize ≤ numobs(data) || return ()
    loader = DataLoader(data; batchsize, shuffle, partial = false)
    return Iterators.map(vd -> (vd, wts), Iterators.cycle(loader))
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
        data, wts = getobs((data, wts), positive_indices)
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

#= Minibatch iteration is delegated to MLUtils.DataLoader. `Iterators.cycle`
restarts the loader whenever an epoch is exhausted, and the loader reshuffles on
each restart, so the stream is infinite with a fresh permutation per epoch.
`partial = false` drops the trailing incomplete batch, so every minibatch holds
exactly `batchsize` observations. A `batchsize` larger than the data clamps to
one full batch (the training entry point clamps before building the iterator,
so the loader's clamping warning is not reached from the trainers). =#
function infinite_minibatches(ds::AbstractArray...; batchsize::Int, shuffle::Bool = true)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    return Iterators.cycle(DataLoader(ds; batchsize, shuffle, partial = false))
end

#= Weights must be finite, positive reals: zero weights are rejected, so
observations meant to be excluded must be dropped (with their weights) before
training. Valid weights are used exactly as given — they are never rescaled
(extreme finite weights can overflow the plain reductions). Non-real weights
are rejected by dispatch; lazy uniform weights are trivially valid. =#
_validate_weights(wts::Ones{<:Real}) = wts

function _validate_weights(wts::AbstractArray{<:Real})
    all(w -> isfinite(w) && w > 0, wts) ||
        throw(ArgumentError("wts must contain only finite, positive values"))
    return wts
end

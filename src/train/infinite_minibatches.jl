"""
    infinite_minibatches(ds...; batchsize, shuffle = true)

Infinite iterator over minibatches of exactly `batchsize` observations, drawn from the
trailing (batch) dimension of each array in `ds`, cycling through the data with a fresh
shuffle on each pass. The trailing incomplete batch of each pass is dropped. A
`batchsize` larger than the data clamps to one full batch (the training entry point
clamps before building the iterator, so the loader's clamping warning is not reached
from the trainers).
"""
function infinite_minibatches(ds::AbstractArray...; batchsize::Int, shuffle::Bool = true)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    return Iterators.cycle(DataLoader(ds; batchsize, shuffle, partial = false))
end

"""
    validate_wts(wts)

Asserts that the data weights `wts` are finite, positive reals, and returns `nothing`.
Zero weights are rejected: observations meant to be excluded must be dropped (with
their weights) before training. Valid weights are used exactly as given by the
training routines — they are never rescaled (extreme finite weights can overflow the
plain reductions). Non-real weights are rejected by dispatch; lazy uniform weights are
trivially valid and skip the check.
"""
validate_wts(::Ones{<:Real}) = nothing

function validate_wts(wts::AbstractArray{<:Real})
    @assert all(w -> isfinite(w) && w > 0, wts) "wts must contain only finite, positive values"
    return nothing
end

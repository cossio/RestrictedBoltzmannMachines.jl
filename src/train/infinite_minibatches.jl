"""
    infinite_minibatches(ds...; batchsize, shuffle = true)

Infinite iterator over shuffled minibatches of `batchsize` observations from `ds`.
"""
function infinite_minibatches(ds::AbstractArray...; batchsize::Int, shuffle::Bool = true)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    return Iterators.cycle(DataLoader(ds; batchsize, shuffle, partial = false))
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

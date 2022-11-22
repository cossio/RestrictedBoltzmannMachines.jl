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
    i = randperm(nobs(ds...))
    return getobs(i, ds...)
end

struct InfiniteMinibatchIterator{T}
    data::T
    batchsize::Int
    shuffle::Bool
end

function Base.iterate(iter::InfiniteMinibatchIterator)
    n = nobs(iter.data...)
    if isnothing(n) || !(0 ≤ iter.batchsize ≤ n)
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
    return InfiniteMinibatchIterator(ds, batchsize, shuffle)
end

export DBM
export inputs_to, inputs_back, inputs_forw
export inputs_odd_to_even, inputs_even_to_odd
export sample_even_from_odd, sample_odd_from_even
export sample_even_from_even, sample_odd_from_odd

"""
    DBM

Deep Boltzmann machine.
"""
struct DBM{L<:Tuple,W<:Tuple}
    layers::L
    weights::W
    function DBM(layers::L, weights::W) where {L<:Tuple, W<:Tuple}
        length(weights) == length(layers) - 1 || dimserror()
        for l = 1:length(layers) - 1
            size(weights[l]) == (size(layers[l])..., size(layers[l+1])...) || dimserror()
        end
        new{L,W}(layers, weights)
    end
end

function DBM(::Type{T}, layers::AbstractLayer...) where {T}
    w = zeros.(T, tuplejoin.(size.(front(layers)), size.(tail(layers))))
    DBM(layers, w)
end

DBM(layers::AbstractLayer...) = DBM(Float64, layers...)

Base.tail(dbm::DBM) = DBM(tail(dbm.layers), tail(dbm.weights))
Base.front(dbm::DBM) = DBM(front(dbm.layers), front(dbm.weights))
tail2(dbm::DBM) = DBM(tail2(dbm.layers), tail2(dbm.weights))
front2(dbm::DBM) = DBM(front2(dbm.layers), front2(dbm.weights))
Base.length(dbm::DBM) = length(dbm.layers)
Base.ndims(dbm::DBM, l::Int) = ndims(dbm.layers[l])

"""
    RBM(dbm, Val(l))

RBM consisting of layers `l` and `l+1` from `dbm`.
"""
RBM(dbm::DBM, ::Val{l}) where {l} = RBM(dbm.layers[l], dbm.layers[l+1], dbm.weights[l])

"""
    Tuple(dbm)

Converts the DBM to a tuple of RBMs.
"""
Base.Tuple(dbm::DBM) = RBM.(front(dbm.layers), tail(dbm.layers), dbm.weights)

function checkdims(dbm::DBM, x::Tuple)
    length(dbm) == length(x) || dimserror()
    checkdims.(dbm.layers, x)
    allequal(batchsize.(dbm.layers, x)...) || dimserror()
end

"""
    energy(dbm, x)

Energy of the DBM in state `x`.
"""
function energy(dbm::DBM, x::Tuple)
    checkdims(dbm, x)
    Ex = sum(map(energy, dbm.layers, x))
    Ew = sum(map(interaction_energy, Tuple(dbm), front(x), tail(x)))
    return Ex + Ew
end

"""
    inputs_forw(dbm, x, Val(l))

Inputs from layer `l` to layer `l+1`.
"""
function inputs_forw(dbm::DBM, x::Tuple, ::Val{l}) where {l}
    checkdims(dbm, x)
    l::Int
    1 ≤ l < length(dbm) || throw(BoundsError(dbm.layers, l))
    tensormul_ff(dbm.weights[l], x[l], Val(ndims(dbm.layers[l])))
end

"""
    inputs_back(dbm, x, Val(l))

Inputs from layer `l+1` to layer `l`.
"""
function inputs_back(dbm::DBM, x::Tuple, ::Val{l}) where {l}
    checkdims(dbm, x)
    l::Int
    1 ≤ l < length(dbm) || throw(BoundsError(dbm.layers, l))
    tensormul_lf(dbm.weights[l], x[l+1], Val(ndims(dbm.layers[l+1])))
end

"""
    inputs_to(dbm, x, Val(l))

Inputs to layer `l`.
"""
function inputs_to(dbm::DBM, x::Tuple, ::Val{l}) where {l}
    l::Int
    1 ≤ l ≤ length(dbm) || throw(BoundsError(dbm.layers, l))
    checkdims(dbm, x)
    if 1 == l < length(dbm)
        inputs_back(dbm, x, Val(l))
    elseif 1 < l == length(dbm)
        inputs_forw(dbm, x, Val(l-1))
    elseif 1 < l < length(dbm)
        inputs_forw(dbm, x, Val(l-1)) + inputs_back(dbm, x, Val(l))
    else
        zero(x[1])
    end
end

"""
    inputs_odd_to_even(dbm, x)

Returns a tuple (nothing, I2, nothing, I4, ...), containing the inputs from
odd layers to even layers in the corresponding even positions.
"""
function inputs_odd_to_even(dbm::DBM, x::Tuple)
    checkdims(dbm, x)
    if length(dbm) == 0
        return ()
    elseif length(dbm) == 1
        return (nothing,)
    elseif length(dbm) == 2
        I2 = tensormul_ff(dbm.weights[1], x[1], Val(ndims(dbm.layers[1])))
        return (nothing, I2)
    else
        If = tensormul_ff(dbm.weights[1], x[1], Val(ndims(dbm.layers[1])))
        Ib = tensormul_lf(dbm.weights[2], x[3], Val(ndims(dbm.layers[3])))
        I2 = If + Ib
        Is = inputs_odd_to_even(tail2(dbm), tail2(x))
        return (nothing, I2, Is...)
    end
end

"""
    inputs_even_to_odd(dbm, x)

Returns a tuple (I1, nothing, I3, nothing, ...), containing the inputs from
even layers to odd layers in the corresponding odd positions.
"""
function inputs_even_to_odd(dbm::DBM, x::Tuple)
    checkdims(dbm, x)
    if length(dbm) == 0
        return ()
    elseif length(dbm) == 1
        return (zero(first(x)),)
    elseif length(dbm) == 2
        I1 = tensormul_lf(dbm.weights[1], x[2], Val(ndims(dbm.layers[2])))
        return (I1, nothing)
    else
        I1 = tensormul_lf(dbm.weights[1], x[2], Val(ndims(dbm.layers[2])))
        If = tensormul_ff(dbm.weights[2], x[2], Val(ndims(dbm.layers[2])))
        Is = inputs_even_to_odd(tail2(dbm), tail2(x))
        return (I1, nothing, If + first(Is), tail(Is)...)
    end
end

"""
    sample_even_from_odd(dbm, x, β=1)

Samples even layers conditioned on the state of odd layers.
"""
function sample_even_from_odd(dbm::DBM, x::Tuple, β=1)
    checkdims(dbm, x)
    I = inputs_odd_to_even(dbm, x)
    ntuple(l -> isodd(l) ? x[l] : random(dbm.layers[l], I[l], β), Val(length(dbm)))
end

"""
    sample_odd_from_even(dbm, x, β=1)

Samples odd layers conditioned on the state of even layers.
"""
function sample_odd_from_even(dbm::DBM, x::Tuple, β=1)
    checkdims(dbm, x)
    I = inputs_even_to_odd(dbm, x)
    ntuple(l -> iseven(l) ? x[l] : random(dbm.layers[l], I[l], β), Val(length(dbm)))
end

"""
    sample_even_from_even(dbm, x, β=1)

Samples odd layers from even layers, then even layers from odd layers.
"""
function sample_even_from_even(dbm::DBM, x::Tuple, β=1)
    length(x) == length(dbm) || dimserror()
    x = sample_odd_from_even(dbm, x, β)
    x = sample_even_from_odd(dbm, x, β)
    return x
end

"""
    sample_odd_from_odd(dbm, x, β=1)

Samples even layers from odd layers, then odd layers from even layers.
"""
function sample_odd_from_odd(dbm::DBM, x::Tuple, β=1)
    length(dbm.layers) == length(x) || dimserror()
    x = sample_even_from_odd(dbm, x, β)
    x = sample_odd_from_even(dbm, x, β)
    return x
end

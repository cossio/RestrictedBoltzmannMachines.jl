struct InputNorm{At,Bt}
    A::At
    b::Bt
    function InputNorm(A::AbstractArray, b::AbstractArray)
        @assert size(A) == (size(b)..., size(b)...)
        return new{typeof(A), typeof(b)}(A, b)
    end
end

Flux.@functor InputNorm

function (layer::InputNorm)(x::AbstractArray)
    @assert size(x) == size(layer.b)
    Amat = reshape(layer.A, length(x), length(x))
    bvec = reshape(layer.b, length(x))
    xvec = reshape(x, length(x))
    out = Amat * xvec + bvec
    return reshape(out, size(x)...)
end

#= As in Bernoulli units, we prefer to encode binary states with
floats 0.0 / 1.0 instead of Booleans, because these hit faster
BLAS linear algebra routines. =#

export Potts
export sitedims, sitesize, siteindices

"""
    Potts{U,N}

Encodes categorical variables as one-hot vectors. The number of classes
is the size of the first dimension.
"""
struct Potts{T,N} <: AbstractLayer{T,N}
    θ::Array{T,N}
end
Potts{T}(n::Int...) where {T} = Potts(zeros(T, n...))
Potts(n::Int...) = Potts{Float64}(n...)
fields(layer::Potts) = (layer.θ,)
Flux.@functor Potts
Base.getproperty(layer::Potts, name::Symbol) =
    name == :q ? size(layer, 1) : getfield(layer, name)
Base.propertynames(::Potts) = (:q, fieldnames(Potts)...)
effective_β(layer::Potts, β) = Potts(β .* layer.θ)
effective_I(layer::Potts, I) = Potts(layer.θ .+ I)
@generated sitedims(layer::Potts{T,N}) where {T,N} = Tuple(2:N)
sitesize(layer::Potts) = Base.tail(size(layer))
siteindices(layer::Potts) = CartesianIndices(OneTo.(sitesize(layer)))
_cgf(layer::Potts) = logsumexp(layer.θ; dims=1)
_transfer_mean(layer::Potts) = OneHot.softmax(layer.θ)
_transfer_mean_abs(layer::Potts) = _transfer_mean(layer)

function energy(layer::Union{Binary,Potts,Spin}, x::AbstractArray)
    checkdims(layer, x)
    E = _binary_energy(layer.θ, x, Val(ndims(layer)))
    return E ./ 1 # convert zero-dim array to number
end

function _random(layer::Potts{T}) where {T}
    X = OneHot.sample_from_logits(layer.θ)
    return T.(X)
end

function _transfer_mode(layer::Potts{T}) where {T}
    classes = OneHot.classify(layer.θ)
    X = OneHot.encode(classes)
    return T.(X)
end

#= gradients =#

_binary_energy(θ, x, ::Val{dims}) where {dims} = -tensormul_ff(θ, x, Val(dims))
@adjoint function _binary_energy(θ, x, ::Val{dims}) where {dims}
    function back(Δ::AbstractArray)
        ∂θ = -tensormul_fl(Δ, x, Val(ndims(x) - dims))
        (∂θ, nothing, nothing)
    end
    back(Δ::Number) = (-Δ .* x, nothing, nothing)
    return _binary_energy(θ, x, Val(dims)), back
end

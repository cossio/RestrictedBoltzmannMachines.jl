#= As in Binary units, we prefer to encode binary states with
floats 0.0 / 1.0 instead of Booleans, because they hit faster
BLAS linear algebra routines. =#

"""
    Potts(θ)

Potts layer, with external fields `θ`.
Encodes categorical variables as one-hot vectors.
The number of classes is the size of the first dimension.
"""
struct Potts{A<:AbstractArray}
    θ::A
end
Potts(::Type{T}, q::Int, n::Int...) where {T} = Potts(zeros(T, q, n...))
Potts(q::Int, n::Int...) = Potts(Float64, q, n...)
Flux.@functor Potts

Base.propertynames(::Potts) = (:q, fieldnames(Potts)...)
function Base.getproperty(layer::Potts, name::Symbol)
    if name == :q
        return size(layer, 1)
    else
        return getfield(layer, name)
    end
end

function cgf(layer::Potts, inputs::AbstractArray)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    Γ = LogExpFunctions.logsumexp(layer.θ .+ inputs; dims = 1)
    return sum_(Γ; dims = layerdims(layer))
end

function cgf(layer::Potts, inputs::AbstractArray, β::Real)
    layer_ = Potts(layer.θ .* β)
    return cgf(layer_, inputs .* β) / β
end

function sample_from_inputs(layer::Potts, inputs::AbstractArray, β::Real)
    layer_ = Potts(layer.θ .* β)
    return sample_from_inputs(layer_, inputs .* β)
end

function sample_from_inputs(layer::Potts, inputs::AbstractArray)
    x = layer.θ .+ inputs
    c = categorical_sample_from_logits(x)
    return oftype(x, onehot_encode(c, 1:layer.q))
end

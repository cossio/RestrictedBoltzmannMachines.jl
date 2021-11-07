struct Binary{A<:AbstractArray}
    θ::A
end
Binary(::Type{T}, n::Int...) where {T} = Binary(zeros(T, n...))
Binary(n::Int...) = Binary(Float64, n...)
Base.size(layer::Binary) = size(layer.θ)
Base.ndims(layer::Binary) = ndims(layer.θ)
Base.length(layer::Binary) = length(layer.θ)

Flux.@functor Binary

function sample_from_inputs(layer::Binary, inputs::AbstractArray)
    @assert size(inputs) == (size(layer.θ)..., size(inputs)[end])
    x = layer.θ .+ inputs
    pinv = @. one(x) + exp(-x)
    u = rand(eltype(pinv), size(pinv))
    return u .* pinv .≤ 1
end

function sample_from_inputs(layer::Binary, inputs::AbstractArray, β::Real)
    @assert size(inputs) == (size(layer.θ)..., size(inputs)[end])
    layer_ = Binary(layer.θ .* β)
    return sample_from_inputs(layer_, inputs .* β)
end

function cgf(layer::Binary, inputs::AbstractArray)
    @assert size(inputs) == (size(layer.θ)..., size(inputs)[end])
    Γ = log1pexp.(layer.θ .+ inputs)
    return sum_(Γ; dims = layerdims(layer))
end

function cgf(layer::Binary, inputs::AbstractArray, β::Real)
    @assert size(inputs) == (size(layer.θ)..., size(inputs)[end])
    layer_ = Binary(layer.θ .* β)
    return cgf(layer_, inputs .* β) / β
end

"""
    Binary(θ)

Binary layer, with external fields `θ`.
"""
struct Binary{A<:AbstractArray}
    θ::A
end
Binary(::Type{T}, n::Int...) where {T} = Binary(zeros(T, n...))
Binary(n::Int...) = Binary(Float64, n...)

Flux.@functor Binary

cgfs(layer::Binary) = LogExpFunctions.log1pexp.(layer.θ)

function transfer_sample(layer::Binary)
    pinv = @. one(layer.θ) + exp(-layer.θ)
    u = rand(eltype(pinv), size(pinv))
    return oftype(layer.θ, u .* pinv .≤ 1)
end

transfer_mean(layer::Binary) = LogExpFunctions.logistic.(layer.θ)

function effective(layer::Binary, inputs, β::Real = 1)
    return Binary(β * (layer.θ .+ inputs))
end

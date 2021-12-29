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
    u = rand(eltype(layer.θ), size(layer))
    return u .* (1 .+ exp.(-layer.θ)) .< 1
end

transfer_mode(layer::Binary) = layer.θ .> 0
transfer_mean(layer::Binary) = LogExpFunctions.logistic.(layer.θ)
transfer_mean_abs(layer::Binary) = transfer_mean(layer)

function transfer_var(layer::Binary)
    return LogExpFunctions.logistic.(layer.θ) .* LogExpFunctions.logistic.(-layer.θ)
end

effective(layer::Binary, inputs, β::Real = true) = Binary(β * (layer.θ .+ inputs))
conjugates(layer::Binary) = (; θ = transfer_mean(layer))

function conjugates_empirical(layer::Binary, samples::AbstractArray)
    @assert size(samples) == (size(layer)..., size(samples)[end])
    μ = mean_(samples; dims=ndims(samples))
    return (; θ = μ)
end

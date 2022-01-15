"""
    Binary(θ)

Binary layer, with external fields `θ`.
"""
struct Binary{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
end
Binary(::Type{T}, n::Int...) where {T} = Binary(zeros(T, n...))
Binary(n::Int...) = Binary(Float64, n...)

Flux.@functor Binary

function effective(layer::Binary, inputs::AbstractArray; β::Real = true)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    return Binary(β * (layer.θ .+ inputs))
end

free_energies(layer::Binary) = -LogExpFunctions.log1pexp.(layer.θ)
transfer_mode(layer::Binary) = layer.θ .> 0
transfer_mean(layer::Binary) = LogExpFunctions.logistic.(layer.θ)
transfer_mean_abs(layer::Binary) = transfer_mean(layer)
transfer_var(layer::Binary) = binary_var.(layer.θ)
transfer_std(layer::Binary) = binary_std.(layer.θ)

function transfer_sample(layer::Binary)
    u = rand(eltype(layer.θ), size(layer))
    return binary_rand.(layer.θ, u)
end

function binary_var(θ::Real)
    t = exp(-abs(θ))
    return t / (1 + t)^2
end

function binary_std(θ::Real)
    t = exp(-abs(θ))
    return √t / (1 + t)
end

function binary_rand(θ::Real, u::Real)
    p = LogExpFunctions.logistic(θ)
    return u < p
end

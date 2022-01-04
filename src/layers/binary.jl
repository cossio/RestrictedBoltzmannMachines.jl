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

free_energies(layer::Binary) = -LogExpFunctions.log1pexp.(layer.θ)

function transfer_sample(layer::Binary)
    u = rand(eltype(layer.θ), size(layer))
    return u .* (1 .+ exp.(-layer.θ)) .< 1
end

transfer_mode(layer::Binary) = layer.θ .> 0
transfer_mean(layer::Binary) = LogExpFunctions.logistic.(layer.θ)
transfer_mean_abs(layer::Binary) = transfer_mean(layer)

function transfer_var(layer::Binary)
    t = @. exp(-abs(layer.θ))
    return @. t / (1 + t)^2
end

effective(layer::Binary, inputs; β::Real = true) = Binary(β * (layer.θ .+ inputs))
∂free_energy(layer::Binary) = (; θ = -transfer_mean(layer))
∂energies(::Binary, x::AbstractArray) = (; θ = -x)

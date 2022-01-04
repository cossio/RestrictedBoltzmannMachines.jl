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
    return map(binary_rand, layer.θ, u)
end

effective(layer::Binary, inputs; β::Real = true) = Binary(β * (layer.θ .+ inputs))
transfer_mode(layer::Binary) = map(binary_mode, layer.θ)
transfer_mean(layer::Binary) = LogExpFunctions.logistic.(layer.θ)
transfer_mean_abs(layer::Binary) = transfer_mean(layer)
transfer_var(layer::Binary) = binary_var.(layer.θ)
∂free_energy(layer::Binary) = (; θ = -transfer_mean(layer))
∂energies(::Binary, x::AbstractArray) = (; θ = -x)

function binary_mode(θ::Real)
    @assert !isnan(θ)
    return θ > 0
end

function binary_var(θ::Real)
    t = exp(-abs(θ))
    return t / (1 + t)^2
end

function binary_rand(θ::Real, u::Real)
    @assert !isnan(θ) && !isnan(u)
    if θ ≥ 0
        t = exp(-θ)
        return u * (1 + t) < 1
    else
        t = exp(θ)
        return u * (1 + t) < t
    end
end

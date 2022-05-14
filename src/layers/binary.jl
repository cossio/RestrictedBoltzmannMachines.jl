"""
    Binary(θ)

Binary layer, with external fields `θ`.
"""
struct Binary{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
end
Binary(::Type{T}, n::Int...) where {T} = Binary(zeros(T, n...))
Binary(n::Int...) = Binary(Float64, n...)

free_energies(layer::Binary, inputs::Union{Real,AbstractArray} = 0) = -log1pexp.(layer.θ .+ inputs)
transfer_mode(layer::Binary, inputs::Union{Real,AbstractArray} = 0) = layer.θ .+ inputs .> 0
transfer_mean(layer::Binary, inputs::Union{Real,AbstractArray} = 0) = logistic.(layer.θ .+ inputs)
transfer_mean_abs(layer::Binary, inputs::Union{Real,AbstractArray} = 0) = transfer_mean(layer, inputs)
var_from_inputs(layer::Binary, inputs::Union{Real,AbstractArray} = 0) = binary_var.(layer.θ .+ inputs)
std_from_inputs(layer::Binary, inputs::Union{Real,AbstractArray} = 0) = binary_std.(layer.θ .+ inputs)

function meanvar_from_inputs(layer::Binary, inputs::Union{Real,AbstractArray} = 0)
    θ = layer.θ .+ inputs
    t = @. exp(-abs(θ))
    μ = @. ifelse(θ ≥ 0, 1 / (1 + t), t / (1 + t))
    ν = @. t / (1 + t)^2
    return μ, ν
end

function transfer_sample(layer::Binary, inputs::Union{Real,AbstractArray} = 0)
    θ = layer.θ .+ inputs
    u = rand!(similar(θ))
    return binary_rand.(θ, u)
end

function binary_var(θ::Real)
    t = exp(-abs(θ))
    return t / (1 + t)^2
end

function binary_std(θ::Real)
    t = exp(-abs(θ) / 2)
    return t / (1 + t^2)
end

binary_rand(θ::Real, u::Real) = u < logistic(θ)

"""
    Binary(θ)

Layer with binary units, with external fields `θ`.
"""
struct Binary{N,A} <: AbstractLayer{N}
    par::A
    function Binary{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 1 # θ
        @assert ndims(par) == N + 1
        return new(par)
    end
end

Binary(par::AbstractArray) = Binary{ndims(par) - 1, typeof(par)}(par)

function Binary(; θ)
    par = vstack((θ,))
    return Binary(par)
end

Binary(::Type{T}, sz::Dims) where {T} = Binary(; θ = zeros(T, sz))
Binary(sz::Dims) = Binary(Float64, sz)

cfgs(layer::Binary, inputs = 0) = -log1pexp.(layer.θ .+ inputs)
mode_from_inputs(layer::Binary, inputs = 0) = layer.θ .+ inputs .> 0
mean_from_inputs(layer::Binary, inputs = 0) = logistic.(layer.θ .+ inputs)
mean_abs_from_inputs(layer::Binary, inputs = 0) = mean_from_inputs(layer, inputs)
var_from_inputs(layer::Binary, inputs = 0) = binary_var.(layer.θ .+ inputs)
std_from_inputs(layer::Binary, inputs = 0) = binary_std.(layer.θ .+ inputs)

function meanvar_from_inputs(layer::Binary, inputs = 0)
    θ = layer.θ .+ inputs
    t = @. exp(-abs(θ))
    μ = @. ifelse(θ ≥ 0, 1 / (1 + t), t / (1 + t))
    ν = @. t / (1 + t)^2
    return μ, ν
end

function sample_from_inputs(layer::Binary, inputs = 0)
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

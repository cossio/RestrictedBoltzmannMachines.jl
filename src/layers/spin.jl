@doc raw"""
    Spin(θ)

Spin layer, with external fields `θ`.
The energy of a layer with units ``s_i`` is given by:

```math
E = -\sum_i \theta_i s_i
```

where each spin ``s_i`` takes values ``\pm 1``.
"""
struct Spin{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
end
Spin(::Type{T}, n::Int...) where {T} = Spin(zeros(T, n...))
Spin(n::Int...) = Spin(Float64, n...)

Base.repeat(l::Spin, n::Int...) = Spin(repeat(l.θ, n...))

free_energies(layer::Spin, inputs::Union{Real,AbstractArray} = 0) = spin_free.(layer.θ .+ inputs)
transfer_mode(layer::Spin, inputs::Union{Real,AbstractArray} = 0) = ifelse.(layer.θ .+ inputs .> 0, Int8(1), Int8(-1))
transfer_mean(layer::Spin, inputs::Union{Real,AbstractArray} = 0) = tanh.(layer.θ .+ inputs)
transfer_mean_abs(layer::Spin, inputs::Union{Real,AbstractArray} = 0) = Ones{Int8}(size(layer))
transfer_std(layer::Spin, inputs::Union{Real,AbstractArray} = 0) = sqrt.(transfer_var(layer, inputs))

function transfer_var(layer::Spin, inputs::Union{Real,AbstractArray} = 0)
    μ = transfer_mean(layer, inputs)
    return @. (1 - μ) * (1 + μ)
end

function meanvar_from_inputs(layer::Spin, inputs::Union{Real,AbstractArray} = 0)
    μ = transfer_mean(layer, inputs)
    ν = @. (1 - μ) * (1 + μ)
    return μ, ν
end

function transfer_sample(layer::Spin, inputs::Union{Real,AbstractArray} = 0)
    θ = layer.θ .+ inputs
    u = rand!(similar(θ))
    return spin_rand.(θ, u)
end

function spin_free(θ::Real)
    abs_θ = abs(θ)
    return -abs_θ - log1pexp(-2abs_θ)
end

spin_rand(θ::Real, u::Real) = ifelse(u < logistic(2θ), Int8(1), Int8(-1))

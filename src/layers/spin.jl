@doc raw"""
    Spin(θ)

Layer with spin units, with external fields `θ`.
The energy of a layer with units ``s_i`` is given by:

```math
E = -\sum_i \theta_i s_i
```

where each spin ``s_i`` takes values ``\pm 1``.
"""
struct Spin{N,A} <: AbstractLayer{N}
    par::A
    function Spin{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 1 # θ
        @assert ndims(par) == N + 1
        return new(par)
    end
end

Spin(par::AbstractArray) = Spin{ndims(par) - 1, typeof(par)}(par)

function Spin(; θ)
    par = vstack((θ,))
    return Spin(par)
end

Spin(::Type{T}, sz::Dims) where {T} = Spin(; θ = zeros(T, sz))
Spin(sz::Dims) = Spin(Float64, sz)

cgfs(layer::Spin, inputs = 0) = spin_cfg.(layer.θ .+ inputs)
mode_from_inputs(layer::Spin, inputs = 0) = ifelse.(layer.θ .+ inputs .> 0, Int8(1), Int8(-1))
mean_from_inputs(layer::Spin, inputs = 0) = tanh.(layer.θ .+ inputs)
mean_abs_from_inputs(layer::Spin, _ = 0) = Ones{Int8}(size(layer))
std_from_inputs(layer::Spin, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))

function var_from_inputs(layer::Spin, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    return @. (1 - μ) * (1 + μ)
end

function meanvar_from_inputs(layer::Spin, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    ν = @. (1 - μ) * (1 + μ)
    return μ, ν
end

function sample_from_inputs(layer::Spin, inputs = 0)
    θ = layer.θ .+ inputs
    u = rand!(similar(θ))
    return spin_rand.(θ, u)
end

function spin_cfg(θ::Real)
    abs_θ = abs(θ)
    return -abs_θ - log1pexp(-2abs_θ)
end

spin_rand(θ::Real, u::Real) = ifelse(u < logistic(2θ), Int8(1), Int8(-1))

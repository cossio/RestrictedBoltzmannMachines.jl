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

Flux.@functor Spin

function effective(layer::Spin, inputs::AbstractArray; β::Real = true)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    return Spin(β * (layer.θ .+ inputs))
end

free_energies(layer::Spin) = spin_free.(layer.θ)
transfer_mode(layer::Spin) = ifelse.(layer.θ .> 0, Int8(1), Int8(-1))
transfer_mean(layer::Spin) = tanh.(layer.θ)
transfer_mean_abs(layer::Spin) = FillArrays.Ones{Int8}(size(layer))
transfer_std(layer::Spin) = sqrt.(transfer_var(layer))

function transfer_var(layer::Spin)
    μ = transfer_mean(layer)
    return @. (1 - μ) * (1 + μ)
end

function transfer_sample(layer::Spin)
    u = rand(eltype(layer.θ), size(layer.θ))
    return spin_rand.(layer.θ, u)
end

function spin_free(θ::Real)
    abs_θ = abs(θ)
    return -abs_θ - LogExpFunctions.log1pexp(-2abs_θ)
end

function spin_rand(θ::Real, u::Real)
    p = LogExpFunctions.logistic(2θ)
    return ifelse(u < p, Int8(1), Int8(-1))
end

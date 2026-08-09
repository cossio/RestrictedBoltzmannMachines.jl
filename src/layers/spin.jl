@doc raw"""
    Spin(θ)

Layer with spin units, with external fields `θ`.
The energy of a layer with units ``s_i`` is given by:

```math
E = -\sum_i \theta_i s_i
```

where each spin ``s_i`` takes values ``\pm 1``.
"""
@declare_layer Spin (θ = zeros,)

cgfs(layer::Spin, inputs::AbstractArray = Falses(size(layer))) = spin_cgf.(layer.θ .+ inputs)
mode_from_inputs(layer::Spin, inputs::AbstractArray = Falses(size(layer))) = ifelse.(layer.θ .+ inputs .> 0, Int8(1), Int8(-1))
mean_from_inputs(layer::Spin, inputs::AbstractArray = Falses(size(layer))) = tanh.(layer.θ .+ inputs)
mean_abs_from_inputs(layer::Spin, _::AbstractArray = Falses(size(layer))) = Ones{Int8}(size(layer))

function var_from_inputs(layer::Spin, inputs::AbstractArray = Falses(size(layer)))
    μ = mean_from_inputs(layer, inputs)
    return @. (1 - μ) * (1 + μ)
end

function meanvar_from_inputs(layer::Spin, inputs::AbstractArray = Falses(size(layer)))
    μ = mean_from_inputs(layer, inputs)
    ν = @. (1 - μ) * (1 + μ)
    return μ, ν
end

function sample_from_inputs(layer::Spin, inputs::AbstractArray = Falses(size(layer)))
    θ = layer.θ .+ inputs
    u = rand!(similar(θ))
    return spin_rand.(θ, u)
end

spin_cgf(θ::Real) = logaddexp(θ, -θ) # log(exp(θ) + exp(-θ))

spin_rand(θ::Real, u::Real) = ifelse(u < logistic(2θ), Int8(1), Int8(-1))

var_from_moments(::Spin, moments::AbstractArray) = (1 .- moments[1, ..]) .* (1 .+ moments[1, ..])

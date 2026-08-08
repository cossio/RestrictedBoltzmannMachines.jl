@doc raw"""
    ReLU(θ, γ)

Layer with ReLU units, with location parameters `θ` and scale parameters `γ`.
The energy of a layer with units ``h_\mu`` is ``E = \sum_\mu U(h_\mu)``, with the
unit potential:

```math
U(h) = \frac{|\gamma|}{2} h^2 - \theta h \qquad (h \ge 0)
```

where ``\theta``, ``\gamma`` are the entries of `θ`, `γ` for the corresponding
unit. Units are constrained to non-negative values (``U(h) = \infty`` for
``h < 0``).
"""
@declare_layer ReLU (θ = zeros, γ = ones)

function energies(layer::ReLU, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return relu_energy.(layer.θ, layer.γ, x)
end

cgfs(layer::ReLU, inputs = 0) = relu_cgf.(layer.θ .+ inputs, layer.γ)
sample_from_inputs(layer::ReLU, inputs = 0) = relu_rand.(layer.θ .+ inputs, layer.γ)
mode_from_inputs(layer::ReLU, inputs = 0) = max.((layer.θ .+ inputs) ./ abs.(layer.γ), 0)
mean_abs_from_inputs(layer::ReLU, inputs = 0) = mean_from_inputs(layer, inputs)

function mean_from_inputs(layer::ReLU, inputs = 0)
    g = Gaussian(layer.par)
    μ = mean_from_inputs(g, inputs)
    σ = sqrt.(var_from_inputs(g, inputs))
    return @. μ + σ * tnmean(-μ / σ)
end

function var_from_inputs(layer::ReLU, inputs = 0)
    g = Gaussian(layer.par)
    μ = mean_from_inputs(g, inputs)
    ν = var_from_inputs(g, inputs)
    return @. ν * tnvar(-μ / √ν)
end

function meanvar_from_inputs(layer::ReLU, inputs = 0)
    g = Gaussian(layer.par)
    μ = mean_from_inputs(g, inputs)
    ν = var_from_inputs(g, inputs)
    σ = sqrt.(ν)
    tμ, tν = tnmeanvar(-μ ./ σ)
    return μ + σ .* tμ, ν .* tν
end

function moments_from_inputs(layer::ReLU, inputs = 0)
    μ, ν = meanvar_from_inputs(layer, inputs)
    return stack([μ, ν .+ μ .^ 2]; dims = 1)
end

function ∂energy_from_moments(layer::ReLU, moments::AbstractArray)
    return ∂energy_from_moments(Gaussian(layer.par), moments)
end

"""
    moments_from_samples(layer::ReLU, data; wts = uniform_weights(layer, data))

Two moment slots: `<x>` and `<x^2>` (same as `Gaussian`).
"""
function moments_from_samples(
        layer::ReLU, data::AbstractArray;
        wts::AbstractArray = uniform_weights(layer, data)
    )
    return moments_from_samples(Gaussian(layer.par), data; wts)
end

function relu_energy(θ::Real, γ::Real, x::Real)
    E = gauss_energy(θ, γ, x)
    return x < 0 ? oftype(E, Inf) : E
end

function relu_cgf(θ::Real, γ::Real)
    abs_γ = abs(γ)
    return logerfcx(-θ / √(2abs_γ)) - log(2abs_γ / π) / 2
end

function relu_rand(θ::Real, γ::Real)
    abs_γ = abs(γ)
    μ = θ / abs_γ
    σ = √inv(abs_γ)
    return randnt_half(μ, σ)
end

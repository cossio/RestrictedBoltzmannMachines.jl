@doc raw"""
    Gaussian(θ, γ)

Gaussian layer, with location parameters `θ` and scale parameters `γ`.
The energy of a layer with units ``h_\mu`` is ``E = \sum_\mu U(h_\mu)``, with the
unit potential:

```math
U(h) = \frac{|\gamma|}{2} h^2 - \theta h
```

where ``\theta``, ``\gamma`` are the entries of `θ`, `γ` for the corresponding
unit, and ``h`` takes values in ``\mathbb{R}``.
"""
@declare_layer Gaussian (θ = zeros, γ = ones)

energies(layer::Gaussian, x::AbstractArray) = gauss_energy.(layer.θ, layer.γ, x)
gauss_energy(θ::Real, γ::Real, x::Real) = (abs(γ) * x / 2 - θ) * x

cgfs(layer::Gaussian, inputs::AbstractArray = Falses(size(layer))) = gauss_cgf.(layer.θ .+ inputs, layer.γ)
gauss_cgf(θ::Real, γ::Real) = θ^2 / abs(2γ) - log(abs(γ) / π / 2) / 2

function sample_from_inputs(layer::Gaussian, inputs::AbstractArray = Falses(size(layer)))
    μ = mean_from_inputs(layer, inputs)
    σ = std_from_inputs(layer, inputs)
    z = randn!(similar(μ))
    return μ .+ σ .* z
end

mean_from_inputs(l::Gaussian, inputs::AbstractArray = Falses(size(l))) = (l.θ .+ inputs) ./ abs.(l.γ)
var_from_inputs(l::Gaussian, inputs::AbstractArray = Falses(size(l))) = inv.(abs.(l.γ .+ zero(inputs)))
mode_from_inputs(l::Gaussian, inputs::AbstractArray = Falses(size(l))) = mean_from_inputs(l, inputs)

function mean_abs_from_inputs(layer::Gaussian, inputs::AbstractArray = Falses(size(layer)))
    μ = mean_from_inputs(layer, inputs)
    ν = var_from_inputs(layer, inputs)
    return @. √(2ν / π) * exp(-μ^2 / (2ν)) + μ * erf(μ / √(2ν))
end

function moments_from_inputs(layer::Gaussian, inputs::AbstractArray = Falses(size(layer)))
    x1 = mean_from_inputs(layer, inputs)
    x2 = x1 .^ 2 .+ var_from_inputs(layer, inputs)
    return stack([x1, x2]; dims = 1)
end

function ∂energy_from_moments(layer::Gaussian, moments::AbstractArray)
    @assert ntuple(d -> size(moments, d), ndims(layer.par)) == size(layer.par)
    x1 = @view moments[1, ..]
    x2 = @view moments[2, ..]
    ∂θ = -x1
    ∂γ = @. sign(layer.γ) * x2 / 2
    return stack([∂θ, ∂γ]; dims = 1)
end

"""
    moments_from_samples(layer::Gaussian, data; wts = uniform_weights(layer, data))

Two moment slots: `<x>` and `<x^2>`.
"""
function moments_from_samples(
        layer::Gaussian, data::AbstractArray;
        wts::AbstractArray = uniform_weights(layer, data)
    )
    x1 = batchmean(layer, data; wts)
    x2 = batchmean(layer, data .^ 2; wts)
    return stack([x1, x2]; dims = 1)
end

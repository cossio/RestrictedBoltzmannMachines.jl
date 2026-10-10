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

cgfs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = relu_cgf.(layer.θ .+ inputs, layer.γ)
sample_from_inputs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = relu_rand.(layer.θ .+ inputs, layer.γ)
mode_from_inputs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = max.((layer.θ .+ inputs) ./ abs.(layer.γ), 0)
mean_abs_from_inputs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = mean_from_inputs(layer, inputs)

mean_from_inputs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = first.(relu_meanvar.(layer.θ .+ inputs, layer.γ))
var_from_inputs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = last.(relu_meanvar.(layer.θ .+ inputs, layer.γ))
meanvar_from_inputs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = _unzip(relu_meanvar.(layer.θ .+ inputs, layer.γ))
moments_from_inputs(layer::ReLU, inputs::AbstractArray = Falses(size(layer))) = _stack_tuples(relu_moments.(layer.θ .+ inputs, layer.γ))

# the moments interface is shared with Gaussian, which has the same parameters
function ∂energy_from_moments(layer::Union{Gaussian, ReLU}, moments::AbstractArray)
    _check_moments(layer, moments)
    x1 = @view moments[1, ..]
    x2 = @view moments[2, ..]
    ∂θ = -x1
    ∂γ = @. sign(layer.γ) * x2 / 2
    return stack([∂θ, ∂γ]; dims = 1)
end

"""
    moments_from_samples(layer::Union{Gaussian, ReLU}, data; [wts])

Two moment slots: `<x>` and `<x^2>`.
"""
function moments_from_samples(
        layer::Union{Gaussian, ReLU}, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(layer, data)
    )
    x1 = batchmean(layer, data; wts)
    x2 = batchmean(layer, data .^ 2; wts)
    return stack([x1, x2]; dims = 1)
end

function relu_energy(θ::Real, γ::Real, x::Real)
    E = gauss_energy(θ, γ, x)
    return x < 0 ? oftype(E, Inf) : E
end

function relu_cgf(θ::Real, γ::Real)
    abs_γ = abs(γ)
    return logerfcx(-θ / √(2abs_γ)) - log(2abs_γ / π) / 2
end

# A ReLU unit is a Gaussian of mean θ / |γ| and variance 1 / |γ|, truncated to x ≥ 0.
function relu_meanvar(θ::Real, γ::Real)
    μ = θ / abs(γ)
    ν = inv(abs(γ))
    σ = √ν
    tμ, tν = tnmeanvar(-μ / σ)
    return μ + σ * tμ, ν * tν
end

# the two moment slots `<x>`, `<x^2>`
function relu_moments(θ::Real, γ::Real)
    μ, ν = relu_meanvar(θ, γ)
    return μ, μ^2 + ν
end

function relu_rand(θ::Real, γ::Real)
    abs_γ = abs(γ)
    μ = θ / abs_γ
    σ = √inv(abs_γ)
    return randnt_half(μ, σ)
end

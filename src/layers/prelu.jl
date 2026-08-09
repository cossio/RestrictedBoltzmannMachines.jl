@doc raw"""
    pReLU(; θ, γ, Δ, η)

A different parameterization of the `dReLU` layer, with shared scale and
asymmetry ratio.
The energy of a layer with units ``h_\mu`` is ``E = \sum_\mu U(h_\mu)``, with the
unit potential:

```math
U(h) = \begin{cases}
    \frac{|\gamma|}{2(1+\eta)} h^2 - \left(\theta + \frac{\Delta}{1+\eta}\right) h & h \ge 0 \\[4pt]
    \frac{|\gamma|}{2(1-\eta)} h^2 - \left(\theta - \frac{\Delta}{1-\eta}\right) h & h < 0
\end{cases}
```

where ``\theta, \gamma, \Delta, \eta`` are the entries of `θ`, `γ`, `Δ`, `η` for
the corresponding unit. This is the `dReLU` potential with
``\theta^\pm = \theta \pm \Delta / (1 \pm \eta)`` and
``\gamma^\pm = \gamma / (1 \pm \eta)``.

Every value of `η` must be finite and lie strictly inside `(-1, 1)`. For
unconstrained learned asymmetry, use `xReLU` or the fixed-scale `nsReLU` instead.
"""
@declare_layer pReLU (θ = zeros, γ = ones, Δ = zeros, η = zeros)

_valid_prelu_eta(η) = η isa Real && isfinite(η) && -1 < η < 1

function _validate_layer_parameters(layer::pReLU)
    ChainRulesCore.ignore_derivatives() do
        all(_valid_prelu_eta, layer.η) || throw(
            ArgumentError(
                "invalid pReLU.η: all values must be finite and lie in the open " *
                    "interval (-1, 1). Use xReLU or nsReLU for unconstrained learned asymmetry."
            )
        )
    end
    return nothing
end

energies(layer::pReLU, x::AbstractArray) = energies(dReLU(layer), x)
cgfs(layer::pReLU, inputs::AbstractArray = Falses(size(layer))) = cgfs(dReLU(layer), inputs)
sample_from_inputs(layer::pReLU, inputs::AbstractArray = Falses(size(layer))) = sample_from_inputs(dReLU(layer), inputs)
mean_from_inputs(layer::pReLU, inputs::AbstractArray = Falses(size(layer))) = mean_from_inputs(dReLU(layer), inputs)
var_from_inputs(layer::pReLU, inputs::AbstractArray = Falses(size(layer))) = var_from_inputs(dReLU(layer), inputs)
meanvar_from_inputs(layer::pReLU, inputs::AbstractArray = Falses(size(layer))) = meanvar_from_inputs(dReLU(layer), inputs)
mode_from_inputs(layer::pReLU, inputs::AbstractArray = Falses(size(layer))) = mode_from_inputs(dReLU(layer), inputs)
mean_abs_from_inputs(layer::pReLU, inputs::AbstractArray = Falses(size(layer))) = mean_abs_from_inputs(dReLU(layer), inputs)

function ∂energy_from_moments(layer::pReLU, moments::AbstractArray)
    _validate_layer_parameters(layer)
    @assert ntuple(d -> size(moments, d), ndims(layer.par)) == size(layer.par)

    xp1 = @view moments[1, ..]
    xn1 = @view moments[2, ..]
    xp2 = @view moments[3, ..]
    xn2 = @view moments[4, ..]

    ∂θ = -(xp1 + xn1)
    ∂γ = @. sign(layer.γ) * (xp2 / (1 + layer.η) + xn2 / (1 - layer.η)) / 2
    ∂Δ = @. -(xp1 / (1 + layer.η) - xn1 / (1 - layer.η))
    ∂η = @. (
        (-abs(layer.γ) * xp2 / 2 + layer.Δ * xp1) / (1 + layer.η)^2 +
            (abs(layer.γ) * xn2 / 2 + layer.Δ * xn1) / (1 - layer.η)^2
    )

    return stack([∂θ, ∂γ, ∂Δ, ∂η]; dims = 1)
end

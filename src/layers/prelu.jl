"""
    pReLU(; θ, γ, Δ, η)

Parametric ReLU layer, with shared scale and asymmetry ratio. Every value of
`η` must be finite and lie strictly inside `(-1, 1)`. For unconstrained learned
asymmetry, use `xReLU` or the fixed-scale `nsReLU` instead.
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
cgfs(layer::pReLU, inputs = 0) = cgfs(dReLU(layer), inputs)
sample_from_inputs(layer::pReLU, inputs = 0) = sample_from_inputs(dReLU(layer), inputs)
mean_from_inputs(layer::pReLU, inputs = 0) = mean_from_inputs(dReLU(layer), inputs)
var_from_inputs(layer::pReLU, inputs = 0) = var_from_inputs(dReLU(layer), inputs)
meanvar_from_inputs(layer::pReLU, inputs = 0) = meanvar_from_inputs(dReLU(layer), inputs)
mode_from_inputs(layer::pReLU, inputs = 0) = mode_from_inputs(dReLU(layer), inputs)
mean_abs_from_inputs(layer::pReLU, inputs = 0) = mean_abs_from_inputs(dReLU(layer), inputs)

function ∂cgfs(layer::pReLU, inputs = 0)
    (; pp, pn, μp, μn, νp, νn) = _drelu_mixture_moments(dReLU(layer), inputs)
    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2

    ∂θ = @. (pp * μp - pn * μn)
    ∂γ = @. -sign(layer.γ) * (pp * μ2p / (1 + layer.η) + pn * μ2n / (1 - layer.η))
    ∂Δ = @. pp * μp / (1 + layer.η) + pn * μn / (1 - layer.η)
    ∂η = @. -(
        pp * (-abs(layer.γ) * μ2p + layer.Δ * μp) / (1 + layer.η)^2 +
            pn * (abs(layer.γ) * μ2n - layer.Δ * μn) / (1 - layer.η)^2
    )

    return vstack((∂θ, ∂γ, ∂Δ, ∂η))
end

function ∂energy_from_moments(layer::pReLU, moments::AbstractArray)
    _validate_layer_parameters(layer)
    @assert size(layer.par) == size(moments)

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

    return vstack((∂θ, ∂γ, ∂Δ, ∂η))
end

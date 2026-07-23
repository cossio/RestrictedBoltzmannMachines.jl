"""
    xReLU(; θ, γ, Δ, ξ)

Extended ReLU layer, like pReLU but with unbounded asymmetry parameter.
"""
@declare_layer xReLU (θ = zeros, γ = ones, Δ = zeros, ξ = zeros)

energies(layer::xReLU, x::AbstractArray) = energies(dReLU(layer), x)
cgfs(layer::xReLU, inputs = 0) = cgfs(dReLU(layer), inputs)
sample_from_inputs(layer::xReLU, inputs = 0) = sample_from_inputs(dReLU(layer), inputs)
mode_from_inputs(layer::xReLU, inputs = 0) = mode_from_inputs(dReLU(layer), inputs)
mean_from_inputs(layer::xReLU, inputs = 0) = mean_from_inputs(dReLU(layer), inputs)
var_from_inputs(layer::xReLU, inputs = 0) = var_from_inputs(dReLU(layer), inputs)
meanvar_from_inputs(layer::xReLU, inputs = 0) = meanvar_from_inputs(dReLU(layer), inputs)
mean_abs_from_inputs(layer::xReLU, inputs = 0) = mean_abs_from_inputs(dReLU(layer), inputs)

function ∂cgfs(layer::xReLU, inputs = 0)
    (; pp, pn, μp, μn, νp, νn) = _drelu_mixture_moments(dReLU(layer), inputs)
    μ2p = @. νp + μp^2
    μ2n = @. νn + μn^2

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. pp * μp - pn * μn
    ∂γ = @. -(pp * μ2p / (1 + η) + pn * μ2n / (1 - η)) / 2
    ∂γ .*= sign.(layer.γ)
    ∂Δ = @. (pp * μp / (1 + η) + pn * μn / (1 - η))
    abs_γ = abs.(layer.γ)
    ∂ξ = @. -(
        pp * (-abs_γ / 2 * μ2p + layer.Δ * μp) / (1 + layer.ξ + abs(layer.ξ))^2 +
            pn * (abs_γ / 2 * μ2n - layer.Δ * μn) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
end

function ∂energy_from_moments(layer::xReLU, moments::AbstractArray)
    @assert size(layer.par) == size(moments)

    xp1 = moments[1, ..]
    xn1 = moments[2, ..]
    xp2 = moments[3, ..]
    xn2 = moments[4, ..]

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = -(xp1 + xn1)
    ∂γ = @. sign(layer.γ) * (xp2 / (1 + η) + xn2 / (1 - η)) / 2
    ∂Δ = @. -xp1 / (1 + η) + xn1 / (1 - η)
    ∂ξ = @. (
        (-abs(layer.γ) / 2 * xp2 + layer.Δ * xp1) / (1 + layer.ξ + abs(layer.ξ))^2 +
            (abs(layer.γ) / 2 * xn2 + layer.Δ * xn1) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
end
